"""Small utilities extracted from the official model learning ports."""

from .cache import KVCache
from .configuration import PreTrainedConfig, remap_legacy_layer_types
from .masks import build_attention_mask
from .multimodal import MultimodalProcessor
from .linear_attention import GatedDeltaNet
from .outputs import GptOssCausalLMOutput, GptOssModelOutput, MoeCausalLMOutput, Qwen3CausalLMOutput, Qwen3ModelOutput
from .processing import MultiModalData, ProcessorMixin
from .rope import (
    RotaryEmbedding,
    YaRNRotaryEmbedding,
    apply_rotary_pos_emb,
    apply_standard_rotary_pos_emb,
    compute_yarn_parameters,
    repeat_kv,
)
from .training import load_balancing_loss
from .vision import (
    get_vision_attention_seqlens,
    get_vision_cu_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

__all__ = [
    "GptOssCausalLMOutput",
    "GptOssModelOutput",
    "GatedDeltaNet",
    "KVCache",
    "MoeCausalLMOutput",
    "MultiModalData",
    "MultimodalProcessor",
    "PreTrainedConfig",
    "ProcessorMixin",
    "Qwen3CausalLMOutput",
    "Qwen3ModelOutput",
    "RotaryEmbedding",
    "YaRNRotaryEmbedding",
    "apply_rotary_pos_emb",
    "apply_standard_rotary_pos_emb",
    "build_attention_mask",
    "compute_yarn_parameters",
    "load_balancing_loss",
    "get_vision_attention_seqlens",
    "get_vision_cu_seqlens",
    "get_vision_interpolation_indices_and_weights",
    "get_vision_position_ids",
    "repeat_kv",
    "remap_legacy_layer_types",
]
