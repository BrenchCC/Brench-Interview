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
"""Qwen3-VL configuration port without the Transformers configuration runtime."""

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


class Qwen3VLVisionConfig(PreTrainedConfig):
    """Official Qwen3-VL vision configuration fields.

    Parameters:
        **kwargs: Official fields or small toy-model overrides.
    """

    model_type = "qwen3_vl_vision"
    base_config_key = "vision_config"

    depth: int = 27
    hidden_size: int = 1152
    hidden_act: str = "gelu_pytorch_tanh"
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    out_hidden_size: int = 3584
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: list[int] | tuple[int, ...] = (8, 16, 24)
    initializer_range: float = 0.02


class Qwen3VLTextConfig(PreTrainedConfig):
    """Official Qwen3-VL text configuration fields.

    Parameters:
        **kwargs: Official fields or small toy-model overrides.
    """

    model_type = "qwen3_vl_text"
    base_config_key = "text_config"
    default_theta = 500000.0
    ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 128000
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    rope_parameters: dict[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs: Any) -> None:
        """Restore KV-head fallback used by the official Qwen3-VL config.

        Parameters:
            **kwargs: Source-compatible base configuration values.
        """
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


class Qwen3VLConfig(PreTrainedConfig):
    """Combine the official Qwen3-VL text and vision sub-configurations.

    Parameters:
        **kwargs: Official fields or small toy-model overrides.
    """

    model_type = "qwen3_vl"
    sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": Qwen3VLTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict[str, Any] | PreTrainedConfig | None = None
    vision_config: dict[str, Any] | PreTrainedConfig | None = None
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs: Any) -> None:
        """Normalize nested sub-config dictionaries exactly as the source does.

        Parameters:
            **kwargs: Source-compatible base configuration values.
        """
        if isinstance(self.vision_config, dict):
            # old ckpt with incorrect model type -> override manually
            if self.vision_config.get("model_type") == "qwen3_vl":
                self.vision_config["model_type"] = "qwen3_vl_vision"
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        super().__post_init__(**kwargs)


__all__ = ["Qwen3VLConfig", "Qwen3VLTextConfig", "Qwen3VLVisionConfig"]
