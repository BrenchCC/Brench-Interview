"""A minimal per-layer key/value cache for autoregressive decoding."""

import torch


class KVCache:
    """Store attention keys and values for every decoder layer."""

    def __init__(self) -> None:
        """Initialize empty cache dictionaries."""
        self.key_cache: dict[int, torch.Tensor] = {}
        self.value_cache: dict[int, torch.Tensor] = {}

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return cached sequence length for one layer.

        Parameters:
            layer_idx: Decoder-layer index whose cache length is requested.
        """
        if layer_idx not in self.key_cache:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V states and return the complete layer cache.

        Parameters:
            key_states: New keys shaped ``[batch, kv_heads, tokens, head_dim]``.
            value_states: New values shaped ``[batch, kv_heads, tokens, head_dim]``.
            layer_idx: Decoder-layer index that owns these states.
        """
        if layer_idx in self.key_cache:
            key_states = torch.cat([self.key_cache[layer_idx], key_states], dim = -2)
            value_states = torch.cat([self.value_cache[layer_idx], value_states], dim = -2)

        self.key_cache[layer_idx] = key_states
        self.value_cache[layer_idx] = value_states
        return key_states, value_states
