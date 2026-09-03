"""YaRN rotary position embeddings used by GPT-OSS.

Key algorithm comments retain the intent of the Hugging Face implementation.
"""

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    try:
        from ..configuration_gpt_oss import GptOssConfig
    except ImportError:
        from configuration_gpt_oss import GptOssConfig


def compute_yarn_parameters(
    config: "GptOssConfig",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, float]:
    """Compute YaRN inverse frequencies and its attention scaling factor.

    Parameters:
        config: GPT-OSS configuration containing the YaRN parameter dictionary.
        device: Optional device on which the inverse-frequency tensor is created.
    """
    rope_parameters = config.rope_parameters or {}
    rope_type = rope_parameters.get("rope_type", "yarn")
    if rope_type != "yarn":
        raise ValueError("This learning implementation only supports YaRN RoPE")

    base = float(rope_parameters.get("rope_theta", config.default_theta))
    factor = rope_parameters.get("factor")
    original_max_positions = int(rope_parameters["original_max_position_embeddings"])
    if factor is None:
        factor = config.max_position_embeddings / original_max_positions
    factor = float(factor)

    def get_mscale(scale: float, mscale: float = 1.0) -> float:
        """Compute YaRN magnitude scaling for a context extension factor.

        Parameters:
            scale: Context extension factor.
            mscale: Optional YaRN magnitude multiplier.
        """
        if scale <= 1:
            return 1.0
        return 0.1 * mscale * math.log(scale) + 1.0

    attention_factor = rope_parameters.get("attention_factor")
    if attention_factor is None:
        mscale = rope_parameters.get("mscale")
        mscale_all_dim = rope_parameters.get("mscale_all_dim")
        if mscale is not None and mscale_all_dim is not None:
            attention_factor = get_mscale(factor, float(mscale)) / get_mscale(factor, float(mscale_all_dim))
        else:
            attention_factor = get_mscale(factor)

    dim = int(config.head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
    beta_fast = float(rope_parameters.get("beta_fast", 32.0))
    beta_slow = float(rope_parameters.get("beta_slow", 1.0))
    truncate = bool(rope_parameters.get("truncate", True))

    def find_correction_dim(num_rotations: float) -> float:
        """Invert YaRN's rotation count to its rotary-dimension location.

        Parameters:
            num_rotations: Rotation boundary defined by YaRN beta parameters.
        """
        numerator = dim * math.log(original_max_positions / (num_rotations * 2 * math.pi))
        return numerator / (2 * math.log(base))

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0)
    high = min(high, dim - 1)
    if low == high:
        high += 0.001

    # 插值表示缩放 position IDs；外推保留原频率 / Interpolation scales positions, extrapolation keeps base frequencies.
    positions = torch.arange(0, dim, 2, device = device, dtype = torch.float32)
    positional_frequencies = base ** (positions / dim)
    extrapolation = 1.0 / positional_frequencies
    interpolation = 1.0 / (factor * positional_frequencies)
    ramp = torch.clamp((torch.arange(dim // 2, device = device, dtype = torch.float32) - low) / (high - low), 0, 1)
    extrapolation_factor = 1.0 - ramp
    inverse_frequencies = interpolation * (1.0 - extrapolation_factor) + extrapolation * extrapolation_factor
    return inverse_frequencies, float(attention_factor)


class YaRNRotaryEmbedding(nn.Module):
    """Generate YaRN-scaled cosine and sine tensors for attention heads."""

    def __init__(self, config: "GptOssConfig") -> None:
        """Initialize fixed YaRN inverse frequencies from a model configuration.

        Parameters:
            config: GPT-OSS configuration used to derive rotary frequencies.
        """
        super().__init__()
        inverse_frequencies, attention_scaling = compute_yarn_parameters(config)
        self.register_buffer("inv_freq", inverse_frequencies, persistent = False)
        self.attention_scaling = attention_scaling

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create YaRN cosine and sine tensors for token positions.

        Parameters:
            hidden_states: Hidden states that provide target dtype and device.
            position_ids: Absolute token positions shaped ``[batch, sequence]``.
        """
        # Force float32 / 原版强制使用 float32 计算频率，避免低精度位置误差。
        inverse_frequencies = self.inv_freq.to(device = hidden_states.device, dtype = torch.float32)
        frequencies = torch.einsum("d,bs->bsd", inverse_frequencies, position_ids.to(torch.float32))
        return (
            (frequencies.cos() * self.attention_scaling).to(dtype = hidden_states.dtype),
            (frequencies.sin() * self.attention_scaling).to(dtype = hidden_states.dtype),
        )


class RotaryEmbedding(nn.Module):
    """Generate standard RoPE cosine and sine tensors for Qwen3 attention."""

    def __init__(self, config: object) -> None:
        """Initialize fixed standard RoPE frequencies from a model configuration.

        Parameters:
            config: Configuration exposing ``head_dim`` and RoPE base parameters.
        """
        super().__init__()
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        base = float(
            rope_parameters.get(
                "rope_theta",
                getattr(config, "default_theta", 1000000.0),
            )
        )
        head_dim = int(getattr(config, "head_dim"))
        inverse_frequencies = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype = torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inverse_frequencies, persistent = False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create standard RoPE cosine and sine tensors for token positions.

        Parameters:
            hidden_states: Hidden states that provide target dtype and device.
            position_ids: Absolute token positions shaped ``[batch, sequence]``.
        """
        # Force float32 / 原版强制使用 float32，避免低精度位置计算误差。
        inverse_frequencies = self.inv_freq.to(device = hidden_states.device, dtype = torch.float32)
        frequencies = torch.einsum("d,bs->bsd", inverse_frequencies, position_ids.to(torch.float32))
        embeddings = torch.cat([frequencies, frequencies], dim = -1)
        return embeddings.cos().to(dtype = hidden_states.dtype), embeddings.sin().to(dtype = hidden_states.dtype)


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply GPT-OSS half-split rotary position embedding to Q and K states.

    Parameters:
        query_states: Query tensor shaped ``[batch, heads, tokens, head_dim]``.
        key_states: Key tensor shaped ``[batch, kv_heads, tokens, head_dim]``.
        cos: YaRN cosine tensor shaped ``[batch, tokens, head_dim / 2]``.
        sin: YaRN sine tensor shaped ``[batch, tokens, head_dim / 2]``.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate(states: torch.Tensor) -> torch.Tensor:
        """Rotate a half-split tensor by the supplied sine and cosine values.

        Parameters:
            states: Attention states with an even-sized final dimension.
        """
        first_half, second_half = torch.chunk(states, 2, dim = -1)
        return torch.cat(
            [first_half * cos - second_half * sin, second_half * cos + first_half * sin],
            dim = -1,
        )

    return rotate(query_states), rotate(key_states)


def apply_standard_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply standard Qwen3 RoPE using the rotate-half convention.

    Parameters:
        query_states: Query tensor shaped ``[batch, heads, tokens, head_dim]``.
        key_states: Key tensor shaped ``[batch, kv_heads, tokens, head_dim]``.
        cos: RoPE cosine tensor shaped ``[batch, tokens, head_dim]``.
        sin: RoPE sine tensor shaped ``[batch, tokens, head_dim]``.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(states: torch.Tensor) -> torch.Tensor:
        """Rotate the two halves of the final dimension.

        Parameters:
            states: Attention states with an even-sized final dimension.
        """
        first_half, second_half = torch.chunk(states, 2, dim = -1)
        return torch.cat([-second_half, first_half], dim = -1)

    return (
        query_states * cos + rotate_half(query_states) * sin,
        key_states * cos + rotate_half(key_states) * sin,
    )


def repeat_kv(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """Repeat key/value heads to match the number of query heads.

    Parameters:
        hidden_states: K/V tensor shaped ``[batch, kv_heads, tokens, head_dim]``.
        num_key_value_groups: Number of query-head groups per K/V head.
    """
    if num_key_value_groups == 1:
        return hidden_states
    return hidden_states.repeat_interleave(num_key_value_groups, dim = 1)
