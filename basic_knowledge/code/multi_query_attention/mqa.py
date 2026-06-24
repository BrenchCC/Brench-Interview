import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head):
        """Initialize multi-query attention.

        Args:
            hidden_dim: The input and output hidden dimension.
            nums_head: The number of query heads.
        """
        super().__init__()
        assert hidden_dim % nums_head == 0, "hidden_dim must be divisible by nums_head"

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = 1
        self.head_dim = hidden_dim // nums_head

        self.query_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)
        self.key_proj = nn.Linear(hidden_dim, self.nums_key_value_head * self.head_dim)
        self.value_proj = nn.Linear(hidden_dim, self.nums_key_value_head * self.head_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, attention_mask = None):
        """Run multi-query attention on the input sequence.

        Args:
            x: The input tensor with shape (batch_size, seq_length, hidden_dim).
            attention_mask: Optional mask broadcastable to (batch_size, nums_head, seq_length, seq_length).
        """
        batch_size, seq_length, hidden_dim = x.size()
        assert hidden_dim == self.hidden_dim, "input hidden_dim must match module hidden_dim"

        # Project queries to multiple heads, but keys and values to one shared head.
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Shape query as (batch_size, nums_head, seq_length, head_dim).
        query = query.view(batch_size, seq_length, self.nums_head, self.head_dim).transpose(1, 2)

        # Shape key/value as (batch_size, 1, seq_length, head_dim).
        key = key.view(batch_size, seq_length, self.nums_key_value_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.nums_key_value_head, self.head_dim).transpose(1, 2)

        # Repeat the single key/value head so each query head can use standard attention.
        key = key.repeat_interleave(self.nums_head, dim = 1)
        value = value.repeat_interleave(self.nums_head, dim = 1)

        attention_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask == 0, float("-1e20"))

        attention_weights = torch.softmax(attention_weights, dim = -1)
        output = attention_weights @ value

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_length, -1)
        output = self.out_proj(output)

        return output


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    attention_mask = None

    X = torch.randn(3, 2, 128)
    net = MultiQueryAttention(hidden_dim = 128, nums_head = 8)
    output = net(X, attention_mask)
    logger.info(f"The output shape is {output.shape}")
