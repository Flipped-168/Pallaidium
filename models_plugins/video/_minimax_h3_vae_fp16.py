# Vendored from https://huggingface.co/OzzyGT/MiniMax_H3_sdnq_4bit_pruned/blob/07b1c077543ba8d7b01d06a80974e1025f47edcf/vae/modeling_minimax_h3_vae_fp16.py
# Bundled so MiniMax H3 loads without trust_remote_code (no per-load code
# fetch, no silent upstream changes). Keep in sync manually if the repo changes.

"""MiniMax-H3 video VAE that holds the float16 decoder this checkpoint was saved with.

`AutoencoderKLMiniMaxH3` pins every top-level module to float32 through `_keep_in_fp32_modules`,
so a pipeline-level `dtype=` cannot reach the weights. That is deliberate upstream: the decode
block runs `vae.decode` inside `torch.autocast(torch.float16)`, and the pin stops a pipeline
running in bfloat16 from downcasting the decoder to a format with fewer mantissa bits than the
recipe is built on.

The cost is that a checkpoint already storing a float16 decoder gets upcast on load: 5.19 GiB of
weights become 9.70 GiB, and the autocast then builds a float16 copy of all of them on every pass,
so decode peaks at 14.90 GiB instead of 6.65 GiB.

This class carries the checkpoint's own layout instead of the caller's dtype: the decoder stays
float16 with its per-block `scale1` / `scale2` modulation tensors in float32 -- exactly the 511/72
split stored in the file -- whatever `dtype` the pipeline passes, including bfloat16.

**It narrows the pin rather than keeping it, and nothing is converted at all.** The entries below
are the name *components* that select the encoder, the quant convs and every per-block anchor, so
those load as float32 while the decoder's bulk loads as the float16 it is stored as. Measured
against widening everything to float32 and narrowing the decoder back: byte-identical weights,
**0.76 GiB of host memory at load instead of 14.35 GiB**, and identical VRAM.

Two details make or break this:

* `_keep_in_fp32_modules` entries must be **name components, not parameter names**. The loader
  matches `module_to_keep_in_fp32 in param_name.split(".")`
  (`diffusers/models/model_loading_utils.load_model_dict_into_meta`), so `"encoder"` pins
  `encoder.conv_in.weight` while `"encoder.conv_in.weight"` pins nothing whatsoever. Diffusers has
  a second, *substring* rule at `modeling_utils.get_parameter_dtype`, but that one only reports a
  dtype -- it never places a weight. A full-name pin therefore fails silently: the float32 tensors
  load as float16 and are widened back afterwards, which does not recover the bits they dropped.
* the **load dtype is forced**, not taken from the caller. With a narrowed pin, a
  `dtype=torch.bfloat16` load lands the decoder in bfloat16, and bfloat16 keeps 8 mantissa bits
  against float16's 10 -- a loss no later cast can undo. Forcing float16 is what makes "the
  checkpoint's own layout whatever the pipeline passes" still true.
"""

import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3


class MiniMaxH3Fp16DecoderVae(AutoencoderKLMiniMaxH3):
    # Name components, not parameter names -- see the module docstring. These select the encoder,
    # both quant convs and every `transformer_blocks.*.scale1` / `scale2` anchor, and nothing else.
    _keep_in_fp32_modules = ["encoder", "quant_conv", "post_quant_conv", "scale1", "scale2"]

    @classmethod
    def extract_init_dict(cls, config_dict, **kwargs):
        """Drop `auto_map` before the config is matched against `__init__`.

        `config.json` carries `auto_map` so `AutoModel` can resolve this class through
        `trust_remote_code`, and diffusers writes that entry itself in `ConfigMixin.to_json_string`.
        Its loader drops `_`-prefixed keys and `quantization_config` before matching the config
        against `__init__` but not `auto_map`, so every load warns about a key diffusers put there.
        The pruned transformer fixes this by naming `auto_map` in its own `__init__`; that is not
        worth replicating this VAE's 22-parameter parent signature for, so drop the key instead.
        """
        config_dict = {k: v for k, v in config_dict.items() if k != "auto_map"}
        return super().extract_init_dict(config_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load the checkpoint's own dtype layout, ignoring the dtype the caller asked for.

        The decoder is stored float16 and must *load* float16: narrowing it from a wider request
        afterwards would cost mantissa bits that are already gone.
        """
        kwargs["torch_dtype"] = torch.float16
        kwargs.pop("dtype", None)
        return super().from_pretrained(*args, **kwargs)
