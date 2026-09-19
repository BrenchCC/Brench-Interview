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
"""GLM-4-MoE model configuration."""

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


class Glm4MoeConfig(PreTrainedConfig):
    """Official GLM-4-MoE configuration fields and backwards-compatible aliases.

    Parameters:
        **kwargs: Official configuration fields or small toy-model overrides.
    """

    model_type = "glm4_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "n_routed_experts",
        "num_mtp_layers": "num_nextn_predict_layers",
    }

    vocab_size: int = 151552
    hidden_size: int = 4096
    intermediate_size: int = 10944
    num_hidden_layers: int = 46
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: dict[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    moe_intermediate_size: int = 1408
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 1
    norm_topk_prob: bool = True
    use_qk_norm: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    num_mtp_layers: int = 1

    def __post_init__(self, **kwargs: Any) -> None:
        """Apply the official partial-RoPE backward-compatibility default.

        Parameters:
            **kwargs: Source-compatible base configuration values.
        """
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": kwargs["partial_rotary_factor"],
            }
        super().__post_init__(**kwargs)


__all__ = ["Glm4MoeConfig"]
