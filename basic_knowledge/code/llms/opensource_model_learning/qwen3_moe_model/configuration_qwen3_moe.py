# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""Qwen3MoE model configuration."""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.append(os.getcwd())

try:
    from ..utils.configuration import PreTrainedConfig
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.configuration import PreTrainedConfig


class Qwen3MoeConfig(PreTrainedConfig):
    """Official Qwen3-MoE configuration fields and sparse-layer scheduling.

    Parameters:
        **kwargs: Official fields or small toy-model overrides.
    """

    model_type = "qwen3_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_experts": "num_local_experts"}

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: dict[str, Any] | None = None
    attention_bias: bool = False
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    attention_dropout: float | int = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 768
    num_experts_per_tok: int = 8
    num_local_experts: int = 128
    norm_topk_prob: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list[int] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs: Any) -> None:
        """Apply source defaults and the local attention-port head-dimension bridge.

        Parameters:
            **kwargs: Source-compatible base configuration values.
        """
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers
        # The current local Qwen3 attention port exposes head_dim explicitly;
        # the upstream implementation derives the identical value inside attention.
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.layer_types = [
            "sliding_attention" if self.sliding_window is not None else "full_attention"
            for _ in range(self.num_hidden_layers)
        ]
        if self.rope_parameters is None:
            self.rope_parameters = {"rope_theta": 1000000.0}
        super().__post_init__(**kwargs)


__all__ = ["Qwen3MoeConfig"]
