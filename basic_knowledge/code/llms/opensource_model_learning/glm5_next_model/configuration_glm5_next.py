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
"""GLM5-Next model configuration."""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.append(os.getcwd())

from ..utils.configuration import PreTrainedConfig


class Glm5NextTextConfig(PreTrainedConfig):
    """Official GLM5-Next text configuration including KDA, DSA, MoE, and mHC.

    Parameters:
        **kwargs: Official fields or small toy-model overrides.
    """

    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}
    vocab_size: int = 154880
    hidden_size: int = 4096
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 45
    num_attention_heads: int = 64
    num_key_value_heads: int | None = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 288
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 1536
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    qk_nope_head_dim: int = 256
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    norm_topk_prob: bool = True
    hidden_act: str = "silu"
    max_position_embeddings: int = 1048576
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 154820
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    mlp_layer_types: list[str] | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    head_dim: int = 0
    layer_types: list[str] | None = None
    # `"full"` runs the indexer, `"shared"` reuses the previous full layer's index mask.
    indexer_types: list[str] | None = None
    base_config_key = "text_config"
    swiglu_limit: float = 10.0
    linear_head_dim: int = 128
    linear_num_heads: int = 64
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float | None = -5.0
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_sinkhorn_iters: int = 20
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    index_kpool: int = 16
    index_kpool_always_select_tail: bool = True

    def __post_init__(self, **kwargs: Any) -> None:
        """Apply source layer schedules, legacy KDA conversion, and derived head sizes.

        Parameters:
            **kwargs: Source-compatible configuration values.
        """
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] * min(3, self.num_hidden_layers) + ["sparse"] * (self.num_hidden_layers - 3)
        if self.layer_types is None:
            kda_layers = [index for index in range(self.num_hidden_layers) if index % 4 != 3]
            self.layer_types = [
                "linear_attention" if layer_idx in kda_layers else "deepseek_sparse_attention"
                for layer_idx in range(self.num_hidden_layers)
            ]
        self.layer_types = [
            "deepseek_sparse_attention" if layer_type == "full_attention" else layer_type
            for layer_type in self.layer_types
        ]
        # Per-layer indexer mode: a pattern (e.g. `"FSSF..."`) overrides the freq/offset schedule.
        if self.indexer_types is None:
            pattern = kwargs.get("index_topk_pattern")
            if pattern is not None:
                self.indexer_types = [{"F": "full", "S": "shared"}[char] for char in pattern] if isinstance(pattern, str) else list(pattern)
            else:
                frequency = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                self.indexer_types = [
                    "full" if (max(index - offset + 1, 0) % frequency) == 0 else "shared"
                    for index in range(self.num_hidden_layers)
                ]
        # Convert dict to attributes (if given)
        linear_attn_dict = kwargs.get("linear_attn_config")
        if linear_attn_dict is not None:
            self.linear_head_dim = linear_attn_dict.get("head_dim", self.linear_head_dim)
            self.linear_num_heads = linear_attn_dict.get("num_heads", self.linear_num_heads)
            self.linear_conv_kernel_dim = linear_attn_dict.get("short_conv_kernel_size", self.linear_conv_kernel_dim)
            self.linear_lower_bound = linear_attn_dict.get("gate_lower_bound", self.linear_lower_bound)
            # Additional lower bound logic as per original dict
            if linear_attn_dict.get("safe_gate", True) and self.linear_lower_bound is None:
                self.linear_lower_bound = -5.0
        # NOTE: this forces an intentional override as we have the convention of head_dim being the RoPE based dim
        self.head_dim = self.qk_rope_head_dim
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        super().__post_init__(**kwargs)

    def validate_architecture(self) -> None:
        """Validate the architecture conditions enforced by source ``@strict``.

        Parameters:
            None.
        """
        if self.num_attention_heads != self.num_key_value_heads:
            raise ValueError("GLM5-Next requires num_attention_heads == num_key_value_heads.")
        if self.index_kpool < 1 or self.index_topk % self.index_kpool != 0:
            raise ValueError("index_kpool must be positive and divide index_topk.")
        if self.q_lora_rank is None:
            raise ValueError("For DSA usage in the attention layers, q_lora_rank is strictly required!")
        if self.qk_rope_head_dim > 0:
            raise ValueError("GLM5-Next DSA expects zero RoPE head dimensions.")


class Glm5NextVisionConfig(PreTrainedConfig):
    """Official GLM5-Next vision configuration fields.

    Parameters:
        **kwargs: Official fields or small toy-model overrides.
    """

    model_type = "glm5_next_vision"
    base_config_key = "vision_config"
    depth: int = 24
    hidden_size: int = 1024
    hidden_act: str = "silu"
    attention_bias: bool = True
    attention_dropout: float | int = 0.0
    num_heads: int = 16
    in_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 336
    patch_size: int | list[int] | tuple[int, int] = 14
    rms_norm_eps: float = 1e-5
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    out_hidden_size: int = 1536
    intermediate_size: int = 4096
    initializer_range: float = 0.02
    projection_intermediate_size: int = 10240
    swiglu_limit: float = 10.0


class Glm5NextConfig(PreTrainedConfig):
    """Combine official GLM5-Next text and vision sub-configurations.

    Parameters:
        **kwargs: Official fields or small toy-model overrides.
    """

    model_type = "glm5_next"
    sub_configs = {"vision_config": Glm5NextVisionConfig, "text_config": Glm5NextTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]
    text_config: dict[str, Any] | PreTrainedConfig | None = None
    vision_config: dict[str, Any] | PreTrainedConfig | None = None
    image_token_id: int = 154854
    video_token_id: int = 154855
    image_start_token_id: int = 154830
    image_end_token_id: int = 154831
    video_start_token_id: int = 154832
    video_end_token_id: int = 154833
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs: Any) -> None:
        """Populate nested configs, including flat text-only checkpoint compatibility.

        Parameters:
            **kwargs: Source-compatible top-level text configuration values.
        """
        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            # Flat (text-only) GLM-5.3-Flash checkpoints store the text fields at the
            # top level; forward them so `text_config` is populated for BC.
            self.text_config = self.sub_configs["text_config"](**kwargs)
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        super().__post_init__(**kwargs)


__all__ = ["Glm5NextConfig", "Glm5NextTextConfig", "Glm5NextVisionConfig"]
