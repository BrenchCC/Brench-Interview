"""MoE training losses."""

import torch
from torch.nn import functional as F


def load_balancing_loss(
    router_logits: tuple[torch.Tensor, ...] | None,
    num_experts: int,
    top_k: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the Switch-style auxiliary loss used to balance expert usage.

    Parameters:
        router_logits: Per-layer tensors shaped ``[batch * tokens, num_experts]``.
        num_experts: Total number of local experts.
        top_k: Number of selected experts per token.
        attention_mask: Optional 1/0 token mask used to exclude padding positions.
    """
    if not router_logits:
        device = attention_mask.device if attention_mask is not None else None
        return torch.tensor(0.0, device = device)

    device = router_logits[0].device
    token_assignments = torch.zeros(num_experts, dtype = torch.float32, device = device)
    probability_sums = torch.zeros(num_experts, dtype = torch.float32, device = device)
    total_tokens = torch.tensor(0.0, dtype = torch.float32, device = device)
    flat_mask = None
    if attention_mask is not None:
        flat_mask = attention_mask.reshape(-1).to(device = device, dtype = torch.float32)

    for layer_logits in router_logits:
        probabilities = F.softmax(layer_logits.float(), dim = -1)
        selected_experts = torch.topk(probabilities, top_k, dim = -1).indices
        # 逐层累积而非拼接 / Accumulate per layer so peak memory stays token-by-expert sized.
        if flat_mask is None:
            token_assignments += torch.bincount(
                selected_experts.reshape(-1),
                minlength = num_experts,
            ).float()
            probability_sums += probabilities.sum(dim = 0)
            total_tokens += probabilities.shape[0]
        else:
            token_assignments.scatter_add_(
                0,
                selected_experts.reshape(-1),
                flat_mask.repeat_interleave(top_k),
            )
            probability_sums += (probabilities * flat_mask.unsqueeze(-1)).sum(dim = 0)
            total_tokens += flat_mask.sum()

    if total_tokens.item() == 0:
        return torch.zeros((), device = device)
    tokens_per_expert = token_assignments / total_tokens
    probability_per_expert = probability_sums / total_tokens
    return num_experts * torch.sum(tokens_per_expert * probability_per_expert)
