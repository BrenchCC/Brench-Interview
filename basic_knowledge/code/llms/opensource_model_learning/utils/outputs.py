"""Lightweight output containers used instead of Transformers ModelOutput."""

from dataclasses import dataclass

import torch

from .cache import KVCache


@dataclass
class GptOssModelOutput:
    """Return hidden states, optional cache, and optional router logits."""

    last_hidden_state: torch.Tensor
    past_key_values: KVCache | None = None
    router_logits: tuple[torch.Tensor, ...] | None = None


@dataclass
class GptOssCausalLMOutput:
    """Return causal-LM logits, optional losses, cache, and router diagnostics."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    cross_entropy_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    past_key_values: KVCache | None = None
    router_logits: tuple[torch.Tensor, ...] | None = None


@dataclass
class Qwen3ModelOutput:
    """Return Qwen3 hidden states and an optional KV cache."""

    last_hidden_state: torch.Tensor
    past_key_values: KVCache | None = None


@dataclass
class Qwen3CausalLMOutput:
    """Return Qwen3 causal-LM logits, optional loss, and optional KV cache."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    past_key_values: KVCache | None = None


@dataclass
class MoeCausalLMOutput:
    """Return MoE logits, losses, cache, and optional router diagnostics."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    past_key_values: KVCache | None = None
    router_logits: tuple[torch.Tensor, ...] | None = None
