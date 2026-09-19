# Copyright 2025 The ZhipuAI Inc. team and HuggingFace Inc. team. All rights reserved.
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
"""GLM-4-MoE core architecture port without Transformers runtime services."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
from torch import nn

sys.path.append(os.getcwd())

try:
    from ..utils import KVCache, MoeCausalLMOutput, build_attention_mask, repeat_kv
    from .configuration_glm4_moe import Glm4MoeConfig
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from configuration_glm4_moe import Glm4MoeConfig
    from utils import KVCache, MoeCausalLMOutput, build_attention_mask, repeat_kv


class Glm4MoeRMSNorm(nn.Module):
    """GLM-4-MoE RMSNorm, equivalent to the source T5LayerNorm formulation.

    Parameters:
        hidden_size: Dimension of hidden-state channels.
        eps: Numerical stability constant.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """Create the learnable normalization scale.

        Parameters:
            hidden_size: Dimension of hidden-state channels.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize in float32 and restore the source input dtype.

        Parameters:
            hidden_states: Tensor whose final dimension is ``hidden_size``.
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Glm4MoeRotaryEmbedding(nn.Module):
    """Source-compatible static partial rotary embedding for GLM-4-MoE.

    Parameters:
        config: GLM-4-MoE configuration containing RoPE parameters.
    """

    def __init__(self, config: Glm4MoeConfig) -> None:
        """Build inverse frequencies for GLM's partial rotary dimensions.

        Parameters:
            config: GLM-4-MoE configuration containing RoPE parameters.
        """
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        partial_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        self.rotary_dim = int(self.head_dim * partial_factor)
        self.rotary_dim -= self.rotary_dim % 2
        theta = config.rope_parameters["rope_theta"]
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.rotary_dim, 2, dtype = torch.float32) / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate GLM partial-RoPE cosine and sine tensors.

        Parameters:
            hidden_states: States supplying target device and dtype.
            position_ids: Absolute positions shaped ``[batch, sequence]``.
        """
        freqs = torch.einsum(
            "d,bs->bsd",
            self.inv_freq.to(device = hidden_states.device),
            position_ids.to(torch.float32),
        )
        embeddings = torch.cat((freqs, freqs), dim = -1)
        return embeddings.cos().to(hidden_states.dtype), embeddings.sin().to(hidden_states.dtype)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Rotate the two half dimensions used by the official RoPE formulation.

    Parameters:
        hidden_states: Tensor with an even final rotary dimension.
    """
    first_half = hidden_states[..., : hidden_states.shape[-1] // 2]
    second_half = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim = -1)


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the source GLM partial-RoPE path to query and key heads.

    Parameters:
        query_states: Query states shaped ``[batch, heads, tokens, head_dim]``.
        key_states: Key states shaped ``[batch, kv_heads, tokens, head_dim]``.
        cos: Rotary cosine values shaped ``[batch, tokens, rotary_dim]``.
        sin: Rotary sine values shaped ``[batch, tokens, rotary_dim]``.
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    query_rot, query_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
    key_rot, key_pass = key_states[..., :rotary_dim], key_states[..., rotary_dim:]
    query_embed = query_rot * cos + rotate_half(query_rot) * sin
    key_embed = key_rot * cos + rotate_half(key_rot) * sin
    return torch.cat([query_embed, query_pass], dim = -1), torch.cat([key_embed, key_pass], dim = -1)


class Glm4MoeAttention(nn.Module):
    """GLM grouped-query attention with optional head-local Q/K RMS normalization.

    Parameters:
        config: GLM-4-MoE configuration.
        layer_idx: Decoder-layer index used for the K/V cache.
    """

    def __init__(self, config: Glm4MoeConfig, layer_idx: int) -> None:
        """Create official GLM projection layout and optional Q/K norms.

        Parameters:
            config: GLM-4-MoE configuration.
            layer_idx: Decoder-layer index used for the K/V cache.
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias = config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias = config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias = False)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = Glm4MoeRMSNorm(self.head_dim, eps = config.rms_norm_eps)
            self.k_norm = Glm4MoeRMSNorm(self.head_dim, eps = config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Compute the GLM causal GQA attention output.

        Parameters:
            hidden_states: Input states shaped ``[batch, tokens, hidden_size]``.
            position_embeddings: GLM partial-RoPE cosine and sine tensors.
            attention_mask: Optional valid-key padding mask.
            past_key_values: Optional per-layer K/V cache.
            use_cache: Whether current K/V tensors are appended to the cache.
        """
        batch_size, query_length, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(batch_size, query_length, -1, self.head_dim)
        key_states = self.k_proj(hidden_states).view(batch_size, query_length, -1, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, query_length, -1, self.head_dim)
        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, *position_embeddings)

        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values.get_seq_length(self.layer_idx)
            if use_cache:
                key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        additive_mask = build_attention_mask(
            attention_mask = attention_mask,
            query_length = query_length,
            key_length = key_states.shape[-2],
            past_length = past_length,
            layer_type = "full_attention",
            sliding_window = None,
            device = hidden_states.device,
            dtype = query_states.dtype,
        )
        weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling + additive_mask
        weights = F.softmax(weights, dim = -1, dtype = torch.float32).to(query_states.dtype)
        weights = F.dropout(weights, p = self.config.attention_dropout if self.training else 0.0, training = self.training)
        output = torch.matmul(weights, value_states).transpose(1, 2).reshape(batch_size, query_length, -1).contiguous()
        return self.o_proj(output)


class Glm4MoeMLP(nn.Module):
    """Official GLM SwiGLU MLP used for dense and shared-expert branches.

    Parameters:
        config: GLM-4-MoE configuration.
        intermediate_size: Optional override for shared-expert width.
    """

    def __init__(self, config: Glm4MoeConfig, intermediate_size: int | None = None) -> None:
        """Create GLM's gate, up, and down projections.

        Parameters:
            config: GLM-4-MoE configuration.
            intermediate_size: Optional shared-expert intermediate width.
        """
        super().__init__()
        width = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, width, bias = False)
        self.up_proj = nn.Linear(config.hidden_size, width, bias = False)
        self.down_proj = nn.Linear(width, config.hidden_size, bias = False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the GLM SwiGLU feed-forward mapping.

        Parameters:
            hidden_states: Input tensor shaped ``[..., hidden_size]``.
        """
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Glm4MoeTopkRouter(nn.Module):
    """Official GLM sigmoid router with group selection and correction bias.

    Parameters:
        config: GLM-4-MoE routing configuration.
    """

    def __init__(self, config: Glm4MoeConfig) -> None:
        """Create source router parameters and persistent correction bias.

        Parameters:
            config: GLM-4-MoE routing configuration.
        """
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.routed_scaling_factor = config.routed_scaling_factor
        self.num_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts, dtype = torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select routed experts using the official GLM grouped sigmoid policy.

        Parameters:
            hidden_states: Token states whose trailing dimension is hidden size.
        """
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias
        group_scores = scores_for_choice.view(-1, self.num_group, self.num_experts // self.num_group).topk(2, dim = -1)[0].sum(dim = -1)
        group_idx = torch.topk(group_scores, k = self.topk_group, dim = -1, sorted = False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(-1, self.num_group, self.num_experts // self.num_group).reshape(-1, self.num_experts)
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k = self.top_k, dim = -1, sorted = False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim = -1, keepdim = True) + 1e-20
        return router_logits, topk_weights * self.routed_scaling_factor, topk_indices


class Glm4MoeExperts(nn.Module):
    """Collection of GLM expert weights stored as source-compatible 3D tensors.

    Parameters:
        config: GLM-4-MoE expert configuration.
    """

    def __init__(self, config: Glm4MoeConfig) -> None:
        """Create packed routed-expert gate/up and down projection tensors.

        Parameters:
            config: GLM-4-MoE expert configuration.
        """
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Apply only experts selected by the official router.

        Parameters:
            hidden_states: Flattened token states.
            top_k_index: Selected expert indices per token.
            top_k_weights: Router weights matching selected experts.
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes = self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim = (-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            index = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[index])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[index]).chunk(2, dim = -1)
            current_hidden_states = F.linear(F.silu(gate) * up, self.down_proj[index])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class Glm4MoeMoE(nn.Module):
    """Official mixed-expert module containing routed and shared experts.

    Parameters:
        config: GLM-4-MoE configuration.
    """

    def __init__(self, config: Glm4MoeConfig) -> None:
        """Create router, packed experts, and the shared expert MLP.

        Parameters:
            config: GLM-4-MoE configuration.
        """
        super().__init__()
        self.experts = Glm4MoeExperts(config)
        self.gate = Glm4MoeTopkRouter(config)
        self.shared_experts = Glm4MoeMLP(config, config.moe_intermediate_size * config.n_shared_experts)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine routed expert output with the always-active shared expert.

        Parameters:
            hidden_states: States shaped ``[batch, tokens, hidden_size]``.
        """
        residuals = hidden_states
        original_shape = hidden_states.shape
        router_logits, weights, indices = self.gate(hidden_states)
        routed = self.experts(hidden_states.view(-1, hidden_states.shape[-1]), indices, weights).view(*original_shape)
        return routed + self.shared_experts(residuals), router_logits


class Glm4MoeDecoderLayer(nn.Module):
    """Official GLM pre-norm decoder with a dense prefix and later MoE layers.

    Parameters:
        config: GLM-4-MoE configuration.
        layer_idx: Decoder-layer index.
    """

    def __init__(self, config: Glm4MoeConfig, layer_idx: int) -> None:
        """Create attention and select dense or mixed-expert MLP by layer index.

        Parameters:
            config: GLM-4-MoE configuration.
            layer_idx: Decoder-layer index.
        """
        super().__init__()
        self.self_attn = Glm4MoeAttention(config, layer_idx)
        self.mlp = Glm4MoeMoE(config) if layer_idx >= config.first_k_dense_replace else Glm4MoeMLP(config)
        self.input_layernorm = Glm4MoeRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = Glm4MoeRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.is_moe = isinstance(self.mlp, Glm4MoeMoE)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: KVCache | None,
        use_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply attention then dense or MoE feed-forward residual paths.

        Parameters:
            hidden_states: Decoder input states.
            position_embeddings: GLM partial-RoPE tensors.
            attention_mask: Optional valid-key padding mask.
            past_key_values: Optional K/V cache.
            use_cache: Whether to update the cache.
        """
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask,
            past_key_values,
            use_cache,
        )
        normalized = self.post_attention_layernorm(hidden_states)
        if self.is_moe:
            mlp_output, router_logits = self.mlp(normalized)
            return hidden_states + mlp_output, router_logits
        return hidden_states + self.mlp(normalized), None


class Glm4MoeModel(nn.Module):
    """Framework-independent GLM-4-MoE decoder model.

    Parameters:
        config: GLM-4-MoE configuration.
    """

    def __init__(self, config: Glm4MoeConfig) -> None:
        """Create token embeddings, decoder layers, final norm, and rotary embedding.

        Parameters:
            config: GLM-4-MoE configuration.
        """
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([Glm4MoeDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = Glm4MoeRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.rotary_emb = Glm4MoeRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
    ) -> tuple[torch.Tensor, KVCache | None, tuple[torch.Tensor, ...]]:
        """Return normalized GLM states, cache, and routed-layer logits.

        Parameters:
            input_ids: Token IDs, mutually exclusive with ``inputs_embeds``.
            attention_mask: Optional valid-token mask.
            past_key_values: Optional decoder cache.
            inputs_embeds: Precomputed token embeddings.
            use_cache: Whether to create or update a K/V cache.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = KVCache()
        past_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(past_length, past_length + inputs_embeds.shape[1], device = inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(inputs_embeds.shape[0], -1)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        router_logits = []
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states, layer_router_logits = layer(
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_values,
                use_cache,
            )
            if layer_router_logits is not None:
                router_logits.append(layer_router_logits)
        return self.norm(hidden_states), past_key_values if use_cache else None, tuple(router_logits)


class Glm4MoeForCausalLM(nn.Module):
    """GLM-4-MoE causal LM retaining source attention and MoE tensor layouts.

    Parameters:
        config: GLM-4-MoE configuration.
    """

    def __init__(self, config: Glm4MoeConfig) -> None:
        """Create the GLM decoder and source-compatible untied LM head.

        Parameters:
            config: GLM-4-MoE configuration.
        """
        super().__init__()
        self.config = config
        self.model = Glm4MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Apply official normal initialization to GLM projections and packed experts.

        Parameters:
            module: Module visited by recursive initialization.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean = 0.0, std = self.config.initializer_range)
        elif isinstance(module, Glm4MoeTopkRouter):
            nn.init.normal_(module.weight, mean = 0.0, std = self.config.initializer_range)
            nn.init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Glm4MoeExperts):
            nn.init.normal_(module.gate_up_proj, mean = 0.0, std = self.config.initializer_range)
            nn.init.normal_(module.down_proj, mean = 0.0, std = self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: KVCache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool = False,
    ) -> MoeCausalLMOutput:
        """Return logits, optional CE loss, cache, and requested GLM router diagnostics.

        Parameters:
            input_ids: Token IDs, mutually exclusive with ``inputs_embeds``.
            attention_mask: Optional valid-token mask.
            past_key_values: Optional decoder cache.
            inputs_embeds: Precomputed token embeddings.
            labels: Optional shifted causal-LM labels.
            use_cache: Whether to update the K/V cache.
            output_router_logits: Whether routed-layer logits are returned.
        """
        hidden_states, cache, router_logits = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.config.vocab_size), labels[:, 1:].reshape(-1), ignore_index = -100)
        return MoeCausalLMOutput(
            logits = logits,
            loss = loss,
            past_key_values = cache,
            router_logits = router_logits if output_router_logits else None,
        )


__all__ = ["Glm4MoeForCausalLM", "Glm4MoeModel"]
