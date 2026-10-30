# Vendored from https://huggingface.co/OzzyGT/MiniMax_H3_sdnq_4bit_pruned/blob/07b1c077543ba8d7b01d06a80974e1025f47edcf/transformer/modeling_minimax_h3_pruned.py
# Bundled so MiniMax H3 loads without trust_remote_code (no per-load code
# fetch, no silent upstream changes). Keep in sync manually if the repo changes.

"""MiniMax-H3 transformer with the AdaLN branch pruned to a cached time-embedding curve.

The released 33.1B checkpoint spends ~13.0B parameters on the per-block AdaLN projections
(`[6 * hidden * modalities, time_embed_dim]` = `[96768, 2688]` x 50). Those projections consume
`silu(time_embedder(t))`, which depends on nothing but the timestep, so over a fixed grid of
timesteps the whole family collapses to a low-rank curve:

    silu(e(t)) ~= P @ c(t)      P: [2688, 8] shared basis,  c(t): 8 coefficients

which lets every block store `W @ P` (`[96768, 8]`) instead of `W`, and lets the model drop the
sinusoidal projection and the time-embedder MLP entirely. `c(t)` is tabulated on a 1025-point grid
(`adaln_t_table`) and interpolated at inference. This is the form the MiniMax-H3 README refers to
when it says the AdaLN modulation outputs "can be precomputed and cached ... for inference-only
deployment"; the full weights are what you need for fine-tuning.

Measured against the full checkpoint, the rank-8 basis reproduces the modulation to a mean relative
error of 8e-5 (worst 1.3e-3), which is below the ~4e-3 noise already present in the int8 weights.

Two behavioural differences follow from the factorisation and are load-bearing:
  * no `silu` - the activation is applied when the curve is built, and re-applying it to the
    coefficients would be wrong (silu is elementwise-nonlinear, the factorisation is linear);
  * the AdaLN projections run in float32, matching the precision the curve was fitted in.
"""

import torch
from torch import nn

from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3AdaLayerNormModulation,
    MiniMaxH3AdaLayerNormOut,
    MiniMaxH3Transformer3DModel,
)


class MiniMaxH3CurveTimeEmbedder(nn.Module):
    r"""
    Stands in for `time_proj` + `time_embedder`: maps `t` in `[0, 1]` to its interpolated row of the
    cached curve basis, so the inherited `forward` needs no change.
    """

    def __init__(self, grid: int, rank: int, basis_dim: int = 0):
        super().__init__()
        self.register_buffer("table", torch.zeros(grid, rank, dtype=torch.float32))
        # The basis the fold was performed against: `silu(e(t)) ~= table[i] @ basis`. Nothing in
        # the forward reads it; it is what lets an adapter trained against the released
        # projections be projected onto these coordinates.
        self.register_buffer("basis", torch.zeros(rank, basis_dim, dtype=torch.float32))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        table = self.table
        # Out-of-range timesteps clamp to the ends of the curve rather than wrapping.
        pos = timestep.to(torch.float32).flatten().clamp(0.0, 1.0) * (table.shape[0] - 1)
        # max-clamp keeps t == 1.0 on the last interval instead of indexing past the table.
        lower = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[lower], table[lower + 1], (pos - lower).unsqueeze(1))


class MiniMaxH3CurveAdaLayerNormModulation(MiniMaxH3AdaLayerNormModulation):
    r"""`MiniMaxH3AdaLayerNormModulation` without the `silu`, which the curve has already absorbed."""

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb = self.linear(temb.to(get_parameter_dtype(self.linear)))
        temb = temb.view(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)


class MiniMaxH3CurveAdaLayerNormOut(MiniMaxH3AdaLayerNormOut):
    r"""`MiniMaxH3AdaLayerNormOut` without the `silu`, which the curve has already absorbed."""

    def forward(
        self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.linear(temb.to(get_parameter_dtype(self.linear))).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class MiniMaxH3PrunedTransformer3DModel(MiniMaxH3Transformer3DModel):
    r"""
    [`MiniMaxH3Transformer3DModel`] with the AdaLN branch stored as a cached curve.

    `time_embed_dim` is the rank of the curve basis (8 in the released pruned checkpoint) rather
    than the width of a time embedding, and `adaln_curve_grid` is the number of tabulated points.
    Everything else - attention, feed-forward, patch projections, RoPE - is unchanged, so the
    forward pass, the modular blocks and the checkpoint keys for those parts all carry over.
    """

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 5376,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 8,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        adaln_curve_grid: int = 1025,
        adaln_basis_dim: int = 2688,
        # `config.json` carries `auto_map` so `AutoModel` can resolve this class through
        # `trust_remote_code`. `ConfigMixin.extract_init_dict` drops `_`-prefixed keys and
        # `quantization_config` before matching the config against `__init__`, but not
        # `auto_map`, so it has to be accepted here or every load warns that it is unexpected.
        auto_map: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            num_refiner_layers=num_refiner_layers,
            ffn_dim=ffn_dim,
            in_channels=in_channels,
            audio_in_channels=audio_in_channels,
            patch_size=patch_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            time_embed_hidden_dim=time_embed_hidden_dim,
            time_embed_dim=time_embed_dim,
            rope_freq_dim=rope_freq_dim,
            rope_theta=rope_theta,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
        )

        # `time_embed_dim` already sized every AdaLN `linear` to the curve rank; replace the two
        # timestep modules with the table lookup and drop the `silu` the curve absorbed.
        self.time_proj = nn.Identity()
        self.time_embedder = MiniMaxH3CurveTimeEmbedder(adaln_curve_grid, time_embed_dim, adaln_basis_dim)
        for block in self.transformer_blocks:
            block.adaln_proj.__class__ = MiniMaxH3CurveAdaLayerNormModulation
        self.norm_out.__class__ = MiniMaxH3CurveAdaLayerNormOut
