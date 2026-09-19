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
"""Qwen3.5 Gated Delta Net kernels extracted from the official PyTorch path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class RMSNormGated(nn.Module):
    """Official Qwen3.5 RMSNormGated implementation.

    Parameters:
        hidden_size: Value-head dimension normalized before gating.
        eps: Numerical stability constant.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Create the gated RMS normalization scale.

        Parameters:
            hidden_size: Value-head dimension.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = "silu"

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize before applying the official SiLU gate.

        Parameters:
            hidden_states: Delta-rule output vectors.
            gate: Per-vector gating projection.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim = True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        return (hidden_states * F.silu(gate.to(torch.float32))).to(input_dtype)


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Tunes out hidden states for padding tokens.

    Parameters:
        hidden_states: Input token states.
        attention_mask: Optional two-dimensional boolean token mask.
    """
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def causal_conv1d_fn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Run the official depthwise causal convolution prefill path.

    Parameters:
        hidden_states: Channels-first projected QKV states.
        weight: Depthwise convolution weights without singleton input dimension.
        bias: Optional convolution bias.
        activation: Optional source activation name.
    """
    _, hidden_size, sequence_length = hidden_states.shape
    padding = weight.shape[-1] - 1
    output = F.conv1d(
        hidden_states.to(weight.dtype),
        weight = weight.unsqueeze(1),
        bias = bias,
        padding = padding,
        groups = hidden_size,
    )[:, :, :sequence_length]
    if activation == "silu":
        output = F.silu(output)
    return output.to(hidden_states.dtype)


def causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """Run the official cached depthwise convolution update path.

    Parameters:
        hidden_states: New channels-first projected QKV states.
        conv_state: Cached trailing convolution inputs, updated in-place.
        weight: Depthwise convolution weights without singleton input dimension.
        bias: Optional convolution bias.
        activation: Optional source activation name.
    """
    _, hidden_size, sequence_length = hidden_states.shape
    state_length = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim = -1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_length:])
    output = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding = 0, groups = hidden_size)
    output = output[:, :, -sequence_length:]
    if activation == "silu":
        output = F.silu(output)
    return output.to(hidden_states.dtype)


def l2norm(hidden_states: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Align with the l2norm implementation used by the FLA library.

    Parameters:
        hidden_states: Tensor to normalize.
        dim: Dimension over which squared norm is measured.
        eps: Numerical stability constant.
    """
    return hidden_states * torch.rsqrt((hidden_states * hidden_states).sum(dim = dim, keepdim = True) + eps)


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute the official token-by-token Gated Delta Rule fallback.

    Parameters:
        query: ``[batch, sequence, value_heads, key_head_dim]`` queries.
        key: ``[batch, sequence, value_heads, key_head_dim]`` keys.
        value: ``[batch, sequence, value_heads, value_head_dim]`` values.
        g: Log-space decays per value head.
        beta: Delta-rule learning-rate values per value head.
        initial_state: Optional cached recurrent state.
        output_final_state: Whether to return the terminal recurrent state.
        use_qk_l2norm_in_kernel: Whether source Q/K L2 normalization is enabled.
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, _, key_head_dim = key.shape
    num_value_heads, value_head_dim = value.shape[-2:]
    query, key, value, beta, decay = [
        tensor.transpose(1, 2).to(torch.float32, memory_format = torch.contiguous_format)
        for tensor in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim = -1, eps = 1e-6)
        key = l2norm(key, dim = -1, eps = 1e-6)
    # And always normalize queries by the head dimension
    query = query / (query.shape[-1] ** 0.5)
    if initial_state is None:
        recurrent_state = torch.zeros(
            (batch_size, num_value_heads, key_head_dim, value_head_dim),
            dtype = value.dtype,
            device = value.device,
        )
    else:
        recurrent_state = initial_state.to(value)
    core_attn_out = torch.zeros_like(value)
    for index in range(sequence_length):
        query_t, key_t, value_t = query[:, :, index], key[:, :, index], value[:, :, index]
        # Decay the recurrent state
        recurrent_state = recurrent_state * decay[:, :, index].exp()[..., None, None]
        # Update the recurrent state with the current token
        beta_t = beta[:, :, index].unsqueeze(-1)
        kv_memory = (recurrent_state * key_t.unsqueeze(-1)).sum(dim = -2)
        delta = (value_t - kv_memory) * beta_t
        recurrent_state = recurrent_state + key_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # And use it to compute the attention output for the current token
        core_attn_out[:, :, index] = (recurrent_state * query_t.unsqueeze(-1)).sum(dim = -2)
    final_state = recurrent_state if output_final_state else None
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), final_state


@dataclass
class GatedDeltaCache:
    """Per-layer convolution and recurrent states needed by Gated Delta Net.

    Parameters:
        conv_states: Layer-indexed depthwise convolution histories.
        recurrent_states: Layer-indexed delta-rule memory matrices.
    """

    conv_states: dict[int, torch.Tensor] = field(default_factory = dict)
    recurrent_states: dict[int, torch.Tensor] = field(default_factory = dict)


class GatedDeltaNet(nn.Module):
    """Official Qwen3.5 Gated Delta Net parameter layout and PyTorch recurrence.

    Parameters:
        config: Qwen3.5 text configuration exposing linear-attention fields.
        layer_idx: Decoder-layer index used to address recurrent cache state.
    """

    def __init__(self, config: Any, layer_idx: int) -> None:
        """Create source QKV convolution, decay, gating, and output projections.

        Parameters:
            config: Qwen3.5 text configuration.
            layer_idx: Decoder-layer index.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, bias = False, kernel_size = self.conv_kernel_size, groups = self.conv_dim, padding = self.conv_kernel_size - 1)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0.01, 16)))
        self.norm = RMSNormGated(self.head_v_dim, eps = config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias = False)
        self.layer_type = config.layer_types[layer_idx]
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias = False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias = False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias = False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias = False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: GatedDeltaCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Qwen3.5 linear attention using official projection and recurrence equations.

        Parameters:
            hidden_states: Text hidden states shaped ``[batch, sequence, hidden_size]``.
            cache_params: Optional per-layer convolution and recurrent cache.
            attention_mask: Optional boolean mask that removes padding states.
        """
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, sequence_length, _ = hidden_states.shape
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, sequence_length, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)
        conv_state = None if cache_params is None else cache_params.conv_states.get(self.layer_idx)
        if conv_state is not None:
            mixed_qkv = causal_conv1d_update(mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)
        else:
            projected_qkv = mixed_qkv
            mixed_qkv = causal_conv1d_fn(projected_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation)
            if cache_params is not None:
                cache_params.conv_states[self.layer_idx] = projected_qkv[:, :, -self.conv_kernel_size + 1 :].detach()
        query, key, value = torch.split(mixed_qkv.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim = -1)
        query = query.reshape(batch_size, sequence_length, -1, self.head_k_dim)
        key = key.reshape(batch_size, sequence_length, -1, self.head_k_dim)
        value = value.reshape(batch_size, sequence_length, -1, self.head_v_dim)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim = 2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim = 2)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        initial_state = None if cache_params is None else cache_params.recurrent_states.get(self.layer_idx)
        core_attn_out, final_state = torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g = g,
            beta = beta,
            initial_state = initial_state,
            output_final_state = cache_params is not None,
            use_qk_l2norm_in_kernel = True,
        )
        if cache_params is not None and final_state is not None:
            cache_params.recurrent_states[self.layer_idx] = final_state.detach()
        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core_attn_out.reshape(batch_size, sequence_length, -1))
