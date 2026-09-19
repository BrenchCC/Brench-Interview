# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Official Qwen3-VL vision encoder port without Transformers runtime layers."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
from torch import nn

sys.path.append(os.getcwd())

try:
    from ..utils.vision import (
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )
    from .configuration_qwen3_vl import Qwen3VLVisionConfig
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from configuration_qwen3_vl import Qwen3VLVisionConfig
    from utils.vision import (
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )


@dataclass
class BaseModelOutputWithDeepstackFeatures:
    """Output shape retained by the official Qwen3-VL vision model.

    Parameters:
        last_hidden_state: Unmerged visual patch hidden states.
        pooler_output: Spatially merged features fed to the language model.
        deepstack_features: Merged intermediate vision features.
    """

    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    deepstack_features: list[torch.Tensor] | None = None


class Qwen3VLVisionMLP(nn.Module):
    """Official two-layer visual MLP.

    Parameters:
        config: Qwen3-VL vision configuration.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        """Create visual feed-forward projections.

        Parameters:
            config: Qwen3-VL vision configuration.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias = True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias = True)
        self.act_fn = nn.GELU(approximate = "tanh") if config.hidden_act == "gelu_pytorch_tanh" else nn.GELU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Apply the visual MLP.

        Parameters:
            hidden_state: Packed visual tokens shaped ``[patches, hidden_size]``.
        """
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    """Official temporal-spatial Conv3D patch embedding.

    Parameters:
        config: Qwen3-VL vision configuration.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        """Create the source Conv3D patch projection.

        Parameters:
            config: Qwen3-VL vision configuration.
        """
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size = kernel_size, stride = kernel_size, bias = True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Embed flattened processor patches using the official reshape path.

        Parameters:
            hidden_states: Flattened patches shaped ``[patches, C * T * P * P]``.
        """
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(dtype = target_dtype)).view(-1, self.embed_dim)


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """Official two-axis visual rotary embedding.

    Parameters:
        dim: Rotary frequency dimension per spatial axis.
        theta: RoPE base.
    """

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """Create fixed visual inverse frequencies.

        Parameters:
            dim: Rotary frequency dimension per spatial axis.
            theta: RoPE base.
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype = torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Return flattened two-axis visual rotary phases.

        Parameters:
            position_ids: Spatial positions shaped ``[patches, 2]``.
        """
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class Qwen3VLVisionPatchMerger(nn.Module):
    """Official Qwen3-VL spatial patch merger.

    Parameters:
        config: Qwen3-VL vision configuration.
        use_postshuffle_norm: Whether normalization occurs after patch shuffle.
    """

    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False) -> None:
        """Create merger norm and MLP projections.

        Parameters:
            config: Qwen3-VL vision configuration.
            use_postshuffle_norm: Select the source deepstack merger normalization path.
        """
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps = 1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Merge spatial patch groups with the official pre/post-shuffle norm choice.

        Parameters:
            hidden_states: Packed visual states in spatial-merge-block order.
        """
        hidden_states = self.norm(
            hidden_states.view(-1, self.hidden_size) if self.use_postshuffle_norm else hidden_states
        ).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input.

    Parameters:
        hidden_states: Tensor with an even final dimension.
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
    """Apply visual RoPE in float32 and restore original tensor dtypes.

    Parameters:
        query_states: Query states in packed-head layout.
        key_states: Key states in packed-head layout.
        cos: Visual rotary cosine values.
        sin: Visual rotary sine values.
    """
    original_query_dtype = query_states.dtype
    original_key_dtype = key_states.dtype
    query_states, key_states = query_states.float(), key_states.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    query_embed = query_states * cos + rotate_half(query_states) * sin
    key_embed = key_states * cos + rotate_half(key_states) * sin
    return query_embed.to(original_query_dtype), key_embed.to(original_key_dtype)


class Qwen3VLVisionAttention(nn.Module):
    """Packed variable-length non-causal attention used by the official vision tower.

    Parameters:
        config: Qwen3-VL vision configuration.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        """Create fused QKV and output projections.

        Parameters:
            config: Qwen3-VL vision configuration.
        """
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias = True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Run attention separately over source packed-variable-length segments.

        Parameters:
            hidden_states: Packed visual tokens shaped ``[patches, hidden_size]``.
            cu_seqlens: Cumulative segment boundaries.
            position_embeddings: Visual rotary cosine and sine values.
            max_seqlen: Retained for source-compatible packed-attention calls.
        """
        del max_seqlen
        sequence_length = hidden_states.shape[0]
        query_states, key_states, value_states = self.qkv(hidden_states).reshape(
            sequence_length,
            3,
            self.num_heads,
            -1,
        ).permute(1, 0, 2, 3).unbind(0)
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, *position_embeddings)
        outputs = []
        for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            query = query_states[start:end].transpose(0, 1)
            key = key_states[start:end].transpose(0, 1)
            value = value_states[start:end].transpose(0, 1)
            weights = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
            weights = nn.functional.softmax(weights, dim = -1, dtype = torch.float32).to(query.dtype)
            outputs.append(torch.matmul(weights, value).transpose(0, 1))
        output = torch.cat(outputs, dim = 0).reshape(sequence_length, -1).contiguous()
        return self.proj(output)


class Qwen3VLVisionBlock(nn.Module):
    """Official pre-norm visual attention and MLP residual block.

    Parameters:
        config: Qwen3-VL vision configuration.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        """Create visual layer norms, attention, and MLP.

        Parameters:
            config: Qwen3-VL vision configuration.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps = 1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps = 1e-6)
        self.attn = Qwen3VLVisionAttention(config)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        """Apply official visual attention and MLP residual updates.

        Parameters:
            hidden_states: Packed visual states.
            cu_seqlens: Cumulative attention segment boundaries.
            position_embeddings: Visual rotary cosine and sine values.
            max_seqlen: Retained for source-compatible packed attention.
        """
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens, position_embeddings, max_seqlen)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3VLVisionModel(nn.Module):
    """Official Qwen3-VL vision tower with interpolation and deepstack outputs.

    Parameters:
        config: Qwen3-VL vision configuration.
    """

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        """Create official vision modules and initialize their weights.

        Parameters:
            config: Qwen3-VL vision configuration.
        """
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        # How the (square) learned position grid is resampled to each image's grid.
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        self.interpolation_align_corners = True
        self.interpolation_mode = "bilinear"
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(config) for _ in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm = False)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [Qwen3VLVisionPatchMerger(config, use_postshuffle_norm = True) for _ in config.deepstack_visual_indexes]
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Apply the official normal initialization scale to visual projections.

        Parameters:
            module: Module visited by recursive initialization.
        """
        if isinstance(module, (nn.Linear, nn.Conv3d, nn.Embedding)):
            nn.init.normal_(module.weight, mean = 0.0, std = self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> BaseModelOutputWithDeepstackFeatures:
        """Encode flattened processor patches using the official vision data flow.

        Parameters:
            hidden_states: Flattened patches shaped ``[patches, C * T * P * P]``.
            grid_thw: Per-image/video temporal-height-width patch grids.
        """
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side = self.num_grid_per_side,
            mode = self.interpolation_mode,
            align_corners = self.interpolation_align_corners,
            spatial_merge_size = self.config.spatial_merge_size,
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config)
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = (self.pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)
        hidden_states = hidden_states + position_embeddings.to(hidden_states.dtype)
        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        sequence_length = hidden_states.shape[0]
        embedding = torch.cat((rotary_pos_emb.reshape(sequence_length, -1),) * 2, dim = -1)
        visual_position_embeddings = (embedding.cos(), embedding.sin())
        deepstack_features = []
        for layer_index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, visual_position_embeddings, max_seqlen)
            if layer_index in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_index)
                deepstack_features.append(self.deepstack_merger_list[merger_index](hidden_states))
        merged_hidden_states = self.merger(hidden_states)
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state = hidden_states,
            pooler_output = merged_hidden_states,
            deepstack_features = deepstack_features,
        )


__all__ = [
    "BaseModelOutputWithDeepstackFeatures",
    "Qwen3VLVisionModel",
    "Qwen3VLVisionPatchEmbed",
    "Qwen3VLVisionPatchMerger",
]
