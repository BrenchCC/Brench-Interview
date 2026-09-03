"""Educational recurrent linear attention blocks used by hybrid model examples."""

import torch
from torch import nn


class GatedDeltaNet(nn.Module):
    """A compact causal gated-delta recurrence for learning hybrid attention."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        """Create Q/K/V/decay projections and an output projection.

        Parameters:
            hidden_size: Input and output hidden dimension.
            num_heads: Number of recurrent attention heads.
            head_dim: Per-head key/value dimension.
        """
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias = False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias = False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias = False)
        self.decay_proj = nn.Linear(hidden_size, num_heads, bias = True)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size, bias = False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply a causal recurrent key-value update without quadratic attention.

        Parameters:
            hidden_states: Input tensor shaped ``[batch, tokens, hidden_size]``.
        """
        batch_size, sequence_length, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        decays = torch.sigmoid(self.decay_proj(hidden_states)).transpose(1, 2)
        state = torch.zeros(
            batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
            device = hidden_states.device,
            dtype = hidden_states.dtype,
        )
        outputs = []
        for token_idx in range(sequence_length):
            key = torch.nn.functional.normalize(k[:, token_idx], dim = -1)
            value = v[:, token_idx]
            state = state * decays[:, :, token_idx, None, None]
            state = state + key.unsqueeze(-1) * value.unsqueeze(-2)
            query = torch.nn.functional.normalize(q[:, token_idx], dim = -1)
            outputs.append(torch.matmul(state, query.unsqueeze(-1)).squeeze(-1))
        output = torch.stack(outputs, dim = 1).reshape(batch_size, sequence_length, -1)
        return self.out_proj(output)
