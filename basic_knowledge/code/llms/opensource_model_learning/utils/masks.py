"""Causal and sliding-window attention masks."""

import torch


def build_attention_mask(
    attention_mask: torch.Tensor | None,
    query_length: int,
    key_length: int,
    past_length: int,
    layer_type: str,
    sliding_window: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create an additive attention mask for one decoder layer.

    Parameters:
        attention_mask: Optional 1/0 padding mask shaped ``[batch, key_length]``.
        query_length: Number of newly processed tokens.
        key_length: Number of cached plus newly processed key tokens.
        past_length: Number of key tokens present before this forward call.
        layer_type: Either ``full_attention`` or ``sliding_attention``.
        sliding_window: Maximum visible token count for sliding attention.
        device: Device on which to allocate the mask.
        dtype: Floating-point type used by attention logits.
    """
    query_positions = past_length + torch.arange(query_length, device = device)
    key_positions = torch.arange(key_length, device = device)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

    if layer_type == "sliding_attention" and sliding_window is not None:
        window_start = query_positions.unsqueeze(1) - sliding_window + 1
        allowed = allowed & (key_positions.unsqueeze(0) >= window_start)

    minimum = torch.finfo(dtype).min
    additive_mask = torch.zeros(query_length, key_length, device = device, dtype = dtype)
    additive_mask.masked_fill_(~allowed, minimum)
    additive_mask = additive_mask.unsqueeze(0).unsqueeze(0)

    if attention_mask is None:
        return additive_mask
    if attention_mask.ndim != 2 or attention_mask.shape[-1] < key_length:
        raise ValueError("attention_mask must have shape [batch, cached_plus_current_tokens]")

    valid_tokens = attention_mask[:, -key_length:].to(device = device, dtype = torch.bool)
    padding_mask = torch.zeros(
        valid_tokens.shape[0],
        1,
        1,
        key_length,
        device = device,
        dtype = dtype,
    )
    padding_mask.masked_fill_(~valid_tokens[:, None, None, :], minimum)
    return additive_mask + padding_mask
