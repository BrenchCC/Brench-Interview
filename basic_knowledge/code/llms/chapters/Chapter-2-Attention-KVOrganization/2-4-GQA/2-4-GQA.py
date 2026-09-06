import os
import sys
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_key_value_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"

        self.head_dim = hidden_dim // num_heads
        self.num_groups = num_heads // num_key_value_heads

        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.key_proj = nn.Linear(hidden_dim, self.head_dim * num_key_value_heads, bias = False)
        self.value_proj = nn.Linear(hidden_dim, self.head_dim * num_key_value_heads, bias = False)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)

        self.atention_dropout = nn.Dropout(0.1)

    def forward(self, x, mask = None):
        """
        x: (batch_size, seq_len, hidden_dim)
        mask: (seq_len, seq_len), True=mask, broadcast to batch and num_heads dimensions
        """
        batch_size, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim, "input hidden_dim must match module hidden_dim"

        # Compute Q, K, V
        q = self.query_proj(x)  # (batch_size, seq_len, hidden_dim)
        k = self.key_proj(x)    # (batch_size, seq_len, head_dim
        v = self.value_proj(x)  # (batch_size, seq_len, head_dim * num_key_value_heads)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_key_value_heads, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_key_value_heads, seq_len, head_dim)

        # Repeat each key/value head to match the query head count.
        k = k.repeat_interleave(self.num_groups, dim = 1)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.repeat_interleave(self.num_groups, dim = 1)  # (batch_size, num_heads, seq_len, head_dim)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores.masked_fill(
            mask, float('-inf')
        )

        attention_weights = F.softmax(attention_scores, dim = -1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = self.atention_dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, head_dim)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, hidden_dim)  # (batch_size, seq_len, hidden_dim)

        output = self.output_proj(attention_output)  # (batch_size, seq_len, hidden_dim)

        return output  # (batch_size, seq_len, hidden_dim)

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    # Example usage
    batch_size = 2
    seq_len = 4
    hidden_dim = 16
    num_heads = 4
    num_key_value_heads = 2

    model = GroupedQueryAttention(hidden_dim, num_heads, num_key_value_heads)
    x = torch.rand(batch_size, seq_len, hidden_dim)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype = torch.bool),
        diagonal = 1,
    )

    output = model(x, mask)
    logger.info(output.shape)  # Should be (batch_size, seq_len, hidden_dim)