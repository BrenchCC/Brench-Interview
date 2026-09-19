# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM5-Next vision tower ported from the official PyTorch implementation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append(os.getcwd())

try:
    from ..utils.vision import get_vision_attention_seqlens, get_vision_position_ids
    from .configuration_glm5_next import Glm5NextVisionConfig
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from configuration_glm5_next import Glm5NextVisionConfig
    from utils.vision import get_vision_attention_seqlens, get_vision_position_ids


@dataclass
class Glm5NextVisionOutput:
    """Framework-free output retaining official vision hidden and merged states.

    Parameters:
        last_hidden_state: Post-downsample visual patch states.
        pooler_output: Final projection fed into the language model.
    """

    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor


class Glm5NextVisionMLP(nn.Module):
    """Official GLM5 vision SwiGLU MLP with clamping.

    Parameters:
        config: GLM5-Next vision configuration.
        bias: Whether projections include a bias.
    """

    def __init__(self, config: Glm5NextVisionConfig, bias: bool = True) -> None:
        """Create source visual MLP projections.

        Parameters:
            config: GLM5-Next vision configuration.
            bias: Projection bias flag.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias = bias)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply source clamped SwiGLU activation.

        Parameters:
            hidden_states: Packed visual token states.
        """
        gate = self.gate_proj(hidden_states).clamp(min = None, max = self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(min = -self.swiglu_limit, max = self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class Glm5NextVisionPatchMerger(nn.Module):
    """Source visual projection after spatial downsampling.

    Parameters:
        dim: Input/output visual feature dimension.
        context_dim: SwiGLU intermediate dimension.
        hidden_act: Retained source activation name.
        swiglu_limit: Source clamp magnitude.
        bias: Projection bias flag.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        hidden_act: str,
        swiglu_limit: float,
        bias: bool = False,
    ) -> None:
        """Create official merger projections.

        Parameters:
            dim: Input/output feature dimension.
            context_dim: Intermediate feature dimension.
            hidden_act: Retained for source-compatible construction.
            swiglu_limit: Source clamp magnitude.
            bias: Projection bias flag.
        """
        super().__init__()
        del hidden_act
        self.proj = nn.Linear(dim, dim, bias = bias)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias = bias)
        self.up_proj = nn.Linear(dim, context_dim, bias = bias)
        self.down_proj = nn.Linear(context_dim, dim, bias = bias)
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project, normalize, GELU, then apply clamped SwiGLU.

        Parameters:
            hidden_states: Downsampled visual states.
        """
        hidden_states = F.gelu(self.post_projection_norm(self.proj(hidden_states)))
        gate = self.gate_proj(hidden_states).clamp(min = None, max = self.swiglu_limit)
        up = self.up_proj(hidden_states).clamp(min = -self.swiglu_limit, max = self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class Glm5NextVisionRMSNorm(nn.Module):
    """GLM5 visual RMSNorm, equivalent to T5LayerNorm.

    Parameters:
        hidden_size: Feature dimension.
        eps: Numerical stability constant.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Create learned visual RMS scale.

        Parameters:
            hidden_size: Feature dimension.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize packed visual states in FP32.

        Parameters:
            hidden_states: Input visual states.
        """
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(hidden_states.square().mean(-1, keepdim = True) + self.variance_epsilon)
        return self.weight * hidden_states.to(dtype)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input.

    Parameters:
        hidden_states: Tensor with an even feature dimension.
    """
    first_half = hidden_states[..., : hidden_states.shape[-1] // 2]
    second_half = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim = -1)


def apply_rotary_pos_emb_vision(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply official visual RoPE in float32 then restore input dtype.

    Parameters:
        query_states: Packed visual queries.
        key_states: Packed visual keys.
        cos: Rotary cosine values.
        sin: Rotary sine values.
    """
    query_dtype, key_dtype = query_states.dtype, key_states.dtype
    query_states, key_states = query_states.float(), key_states.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    query_states = (query_states * cos + rotate_half(query_states) * sin).to(query_dtype)
    key_states = (key_states * cos + rotate_half(key_states) * sin).to(key_dtype)
    return query_states, key_states


class Glm5NextVisionAttention(nn.Module):
    """Official packed variable-length, non-causal vision attention.

    Parameters:
        config: GLM5-Next vision configuration.
    """

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        """Create fused QKV, output projections, and per-head RMSNorm.

        Parameters:
            config: GLM5-Next vision configuration.
        """
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias = config.attention_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias = config.attention_bias)
        self.q_norm = Glm5NextVisionRMSNorm(self.head_dim, eps = config.rms_norm_eps)
        self.k_norm = Glm5NextVisionRMSNorm(self.head_dim, eps = config.rms_norm_eps)
        self.scaling = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Apply attention independently to every packed image/video segment.

        Parameters:
            hidden_states: Packed visual states.
            cu_seqlens: Cumulative segment boundaries.
            position_embeddings: Rotary cosine and sine values.
            max_seqlen: Retained for official flash-attention interface compatibility.
        """
        del max_seqlen
        sequence_length = hidden_states.shape[0]
        query, key, value = self.qkv(hidden_states).reshape(sequence_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        query, key = apply_rotary_pos_emb_vision(query, key, *position_embeddings)
        chunks = []
        # Other implementations: process each packed image/video chunk separately.
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            q = query[start:end].transpose(0, 1)
            k = key[start:end].transpose(0, 1)
            v = value[start:end].transpose(0, 1)
            weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
            weights = F.softmax(weights, dim = -1, dtype = torch.float32).to(q.dtype)
            chunks.append(torch.matmul(weights, v).transpose(0, 1))
        return self.proj(torch.cat(chunks, dim = 0).reshape(sequence_length, -1).contiguous())


class Glm5NextVisionBlock(nn.Module):
    """Official visual pre-norm attention/MLP residual block.

    Parameters:
        config: GLM5-Next vision configuration.
    """

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        """Create visual norms, attention, and MLP.

        Parameters:
            config: GLM5-Next vision configuration.
        """
        super().__init__()
        self.norm1 = Glm5NextVisionRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.norm2 = Glm5NextVisionRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.attn = Glm5NextVisionAttention(config)
        self.mlp = Glm5NextVisionMLP(config, bias = config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Apply source visual residual updates.

        Parameters:
            hidden_states: Packed visual states.
            cu_seqlens: Packed sequence boundaries.
            position_embeddings: Visual RoPE values.
            max_seqlen: Retained for source-compatible calls.
        """
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings, max_seqlen)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Glm5NextVisionRotaryEmbedding(nn.Module):
    """Official visual rotary inverse-frequency table.

    Parameters:
        dim: Rotary dimension.
        theta: RoPE base.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """Create fixed source inverse frequencies.

        Parameters:
            dim: Rotary dimension.
            theta: RoPE base.
        """
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype = torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Compute flattened two-axis rotary phases.

        Parameters:
            position_ids: Packed two-axis positions.
        """
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class Glm5NextVisionPatchEmbed(nn.Module):
    """Official temporal-spatial Conv3D patch embedding.

    Parameters:
        config: GLM5-Next vision configuration.
    """

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        """Create source Conv3D patch projection.

        Parameters:
            config: GLM5-Next vision configuration.
        """
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size = kernel, stride = kernel)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Embed flattened processor patches with source reshape semantics.

        Parameters:
            hidden_states: Flattened patches ``[N, C * T * P * P]``.
        """
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(hidden_states.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


class Glm5NextVisionModel(nn.Module):
    """Official GLM5-Next vision tower without Transformers attention backends.

    Parameters:
        config: GLM5-Next vision configuration.
    """

    def __init__(self, config: Glm5NextVisionConfig) -> None:
        """Create patch embedder, visual blocks, downsample, and final merger.

        Parameters:
            config: GLM5-Next vision configuration.
        """
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Glm5NextVisionPatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Glm5NextVisionBlock(config) for _ in range(config.depth)])
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size = config.spatial_merge_size,
            stride = config.spatial_merge_size,
        )
        self.post_layernorm = Glm5NextVisionRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.merger = Glm5NextVisionPatchMerger(
            config.out_hidden_size,
            config.projection_intermediate_size,
            config.hidden_act,
            config.swiglu_limit,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Apply the official normal initialization scale.

        Parameters:
            module: Module visited by recursive initialization.
        """
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            nn.init.normal_(module.weight, mean = 0.0, std = self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, Glm5NextVisionRMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> Glm5NextVisionOutput:
        """Encode packed image/video patches through the official visual path.

        Parameters:
            hidden_states: Flattened processor patches.
            grid_thw: Temporal-height-width grid for each packed media item.
        """
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config)
        hidden_states = self.patch_embed(hidden_states)
        rotary = self.rotary_pos_emb(position_ids)
        rotary = torch.cat((rotary, rotary), dim = -1)
        position_embeddings = (rotary.cos(), rotary.sin())
        for block in self.blocks:
            hidden_states = block(hidden_states, cu_seqlens, position_embeddings, max_seqlen)
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = hidden_states.view(-1, self.spatial_merge_size, self.spatial_merge_size, hidden_states.shape[-1])
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).view(-1, self.config.out_hidden_size)
        return Glm5NextVisionOutput(last_hidden_state = hidden_states, pooler_output = self.merger(hidden_states))


__all__ = ["Glm5NextVisionModel", "Glm5NextVisionOutput"]
