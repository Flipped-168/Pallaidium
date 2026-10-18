"""Text-prompted sound isolation via SAM-Audio large (Meta).

Describe a sound ("a dog barking", "female speech", "piano") and the model
splits the input audio/video strip into that sound (target) and everything
else (residual). Both come back as SOUND strips.

Weights come from the ungated mirror ml-resources/sam-audio-large (same files
as facebook/sam-audio-large). The checkpoint omits the span predictor and the
rankers, which sam_audio would otherwise fetch from facebook/* repos; this
plugin turns them off in the config, so only the mirror is downloaded. The
model runs in bf16 and long clips are separated in 30 s chunks.
"""

import json
import os
import sys

from ...models.base import ModelPlugin, InputSpec, UISection, ParamSpec, ModelInputs
from ...utils.helpers import solve_path
from .stem_split import _to_wav

_REPO        = "ml-resources/sam-audio-large"
_CHUNK_S     = 30.0
_MULTI_STEM_PREFIX = "MULTI_STEM:"


def _install_import_shims():
    """Let sam_audio import on Windows without xformers / a NumPy-2-safe wandb.

    perception_models' core/transformer.py imports xformers at module level but
    only uses it for the "fmha" attention path (the audio encoder uses sdpa),
    and timm pulls in wandb, which crashes on NumPy 2 (np.float_).
    """
    import types

    if "wandb" not in sys.modules:
        sys.modules["wandb"] = None          # makes `import wandb` a clean ImportError
    try:
        import xformers.ops  # noqa: F401
        return
    except Exception:
        pass
    xf, ops, fmha = (types.ModuleType(n) for n in ("xformers", "xformers.ops", "xformers.ops.fmha"))

    class AttentionBias:                     # only used in type annotations / isinstance on the fmha path
        pass

    ops.AttentionBias, ops.fmha, xf.ops = AttentionBias, fmha, ops
    sys.modules.update({"xformers": xf, "xformers.ops": ops, "xformers.ops.fmha": fmha})


class SamAudioPlugin(ModelPlugin):
    MODEL_ID     = _REPO
    DISPLAY_NAME = "Sound Isolation: SAM-Audio Large"
    MODEL_TYPE   = "audio"
    DESCRIPTION  = "Isolate any sound from an audio/video strip with a text description; outputs the sound and the remainder"

    # The audio/video strip is the mixture; the prompt names the sound to isolate.
    INPUTS       = InputSpec.PROMPT | InputSpec.AUDIO_REF
    UI_SECTIONS  = [UISection.PROMPT, UISection.SEED]
    PARAMS       = ParamSpec()
    # sam_audio is not imported by is_available() (needs the shims) -> checked via find_spec.
    REQUIRED_PACKAGES = []

    supports_inpaint     = False
    supports_img2img     = False
    show_enhance         = False
    requires_input_strip = True
    supports_batch       = False   # one deterministic split per seed

    def is_available(self) -> bool:
        import importlib.util as ilu
        return all(ilu.find_spec(m) is not None for m in ("sam_audio", "dacvae", "core", "torchdiffeq"))

    def load(self, prefs, scene, **kw):
        import torch
        from huggingface_hub import snapshot_download

        _install_import_shims()
        from sam_audio import SAMAudio, SAMAudioProcessor

        local = bool(getattr(prefs, "local_files_only", False))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        print(f"Loading SAM-Audio Large on {device} ({dtype})...")
        model_dir = snapshot_download(_REPO, local_files_only=local)

        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as fh:
            cfg_dict = json.load(fh)
        # Not in the checkpoint; would be fetched from facebook/pe-a-frame-large,
        # facebook/sam-audio-judge, CLAP and ImageBind. Spans/reranking are off.
        for key in ("span_predictor", "text_ranker", "visual_ranker"):
            cfg_dict[key] = None
        cfg = SAMAudio.config_cls(**cfg_dict)

        # Build in bf16 on the CPU (the T5 text encoder calls from_pretrained inside the
        # constructor, so no meta/cuda device context). The fp32 checkpoint is mmapped and
        # cast per tensor, so host RAM holds only the bf16 model (~half the checkpoint).
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            model = SAMAudio(cfg)
        finally:
            torch.set_default_dtype(prev_dtype)
        state = torch.load(
            os.path.join(model_dir, "checkpoint.pt"),
            weights_only=True, map_location="cpu", mmap=True,
        )
        model.load_state_dict(state, strict=True)
        del state
        model = model.to(device).eval()

        processor = SAMAudioProcessor.from_pretrained(model_dir)
        return {"pipe": None, "model": model, "vocoder": None, "feature_extractor": None,
                "processor": processor, "device": device, "dtype": dtype}

    def generate(self, pipe_obj, inputs: ModelInputs, scene, prefs) -> str:
        import numpy as np
        import soundfile as sf
        import torch

        model, processor = pipe_obj["model"], pipe_obj["processor"]
        device, dtype = pipe_obj["device"], pipe_obj["dtype"]

        audio_path = inputs.audio_ref
        description = (inputs.prompt or "").strip()
        if not audio_path:
            raise ValueError("No audio reference path — select a SOUND or MOVIE strip as input.")
        if not description:
            raise ValueError("Describe the sound to isolate in the prompt (e.g. 'a dog barking').")

        torch.manual_seed(inputs.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(inputs.seed)

        self.set_phase(inputs, "Extracting audio…")
        wav_path, is_temp = _to_wav(audio_path)
        import torchaudio
        try:
            wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        finally:
            if is_temp:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass
        sample_rate = processor.audio_sampling_rate
        wav = torch.from_numpy(np.ascontiguousarray(wav.mean(axis=1)))        # mono
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)

        chunk = int(_CHUNK_S * sample_rate)
        pieces = [wav[i:i + chunk] for i in range(0, wav.shape[0], chunk)]
        # Fold a tiny tail (<2 s) into the previous chunk instead of separating it alone.
        if len(pieces) > 1 and pieces[-1].shape[0] < 2 * sample_rate:
            pieces[-2:] = [torch.cat(pieces[-2:])]

        print(f"SAM-Audio | prompt={description!r} | {wav.shape[0] / sample_rate:.1f}s in {len(pieces)} chunk(s)")
        self.set_phase(inputs, "Separating…")
        targets, residuals = [], []
        for i, piece in enumerate(pieces):
            batch = processor(descriptions=[description], audios=[piece.unsqueeze(0)]).to(device)
            with torch.inference_mode(), torch.autocast(device, dtype=dtype, enabled=(device == "cuda")):
                result = model.separate(batch, predict_spans=False, reranking_candidates=1)
            targets.append(result.target[0].float().cpu())
            residuals.append(result.residual[0].float().cpu())
            if inputs.progress_fn is not None:
                inputs.progress_fn(i + 1, len(pieces))

        self.set_phase(inputs, "Saving…")
        base = os.path.splitext(os.path.basename(audio_path))[0]
        paths = {}
        for name, parts in (("target", targets), ("residual", residuals)):
            out = solve_path(f"{name}_{base}.wav")
            sf.write(out, torch.cat(parts).numpy(), sample_rate)
            paths[name] = out
        print(f"SAM-Audio saved: {paths}")
        return _MULTI_STEM_PREFIX + json.dumps(paths)
