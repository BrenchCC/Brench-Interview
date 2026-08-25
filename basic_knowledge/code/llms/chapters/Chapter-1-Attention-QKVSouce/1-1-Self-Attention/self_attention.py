import os
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)

        # 一般是 0.1 的dropout，在训练时随机丢弃一些连接，防止过拟合
        self.attention_dropout = nn.Dropout(0.1)

    def forward(self, x, mask = None):
        # x: (batch, seq_len, hidden_dim)
        # Compute Q, K, V 
        q = self.query_proj(x)  # (batch, seq_len, hidden_dim)
        k = self.key_proj(x)    # (batch, seq_len, hidden_dim)
        v = self.value_proj(x)  # (batch, seq_len, hidden_dim)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_dim) 
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        logger.info("Attention scores shape: %s", scores.shape)
        logger.info("Attention scores: %s", scores)

        scores = self.attention_dropout(scores)  # Apply dropout to attention scores

        attention_weights = F.softmax(scores, dim = -1)  # (batch, seq_len, seq_len)
        attention_output = torch.matmul(attention_weights, v)  # (batch, seq_len, hidden_dim)
        output = self.output_proj(attention_output)  # (batch, seq_len, hidden
        return output

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    batch_size = 2
    seq_length = 4
    hidden_dim = 8

    x = torch.randn(batch_size, seq_length, hidden_dim)
    attention = SelfAttention(hidden_dim)
    output = attention(x)
    logger.info("Output shape: %s", output.shape)
    logger.info("Output: %s", output)

    x = torch.randn(batch_size, seq_length, hidden_dim)
    # mask = torch.repeat(torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]), batch_size // 2, 1)  # (batch_size, seq_length)
    mask = torch.triu(torch.ones(seq_length, seq_length, dtype = torch.bool), diagonal = 1)
    output_with_mask = attention(x, mask = mask)
    logger.info("Output with mask shape: %s", output_with_mask.shape)
    logger.info("Output with mask: %s", output_with_mask)
