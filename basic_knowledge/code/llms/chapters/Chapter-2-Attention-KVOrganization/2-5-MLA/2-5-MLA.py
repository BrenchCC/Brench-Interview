import os
import sys
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, latent_dim):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.key_value_down_proj = nn.Linear(hidden_dim, latent_dim, bias = False)
        self.key_up_proj = nn.Linear(latent_dim, hidden_dim, bias = False)
        self.value_up_proj = nn.Linear(latent_dim, hidden_dim, bias = False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)

    def forward(self, x, mask = None, past_latent_kv = None):
        """
        x: (batch, seq_len, hidden_dim)
        mask: (seq_len, past_len + seq_len)
        past_latent_kv: None或(batch, past_len+seq_len, latent_dim)
        """
        batch, seq_len, _ = x.shape
        
        # Compute Q and latent K/V
        q = self.query_proj(x)  # q: (batch, seq_len, hidden_dim)
        latent_kv = self.key_value_down_proj(x)  # latent_kv: (batch, seq_len, latent_dim)

        # using KV Cache
        past_len = past_latent_kv.size(-2) if past_latent_kv is not None else 0
        if past_latent_kv is not None:
            # Concatenate new and old latent_kv along seq_len dimension: (batch, past_len+seq_len, latent_dim)
            latent_kv = torch.cat([past_latent_kv, latent_kv], dim = -2)
        new_latent_kv = latent_kv

        # Restore latent K/V
        k = self.key_up_proj(latent_kv)  # k: (batch, past_len+seq_len, hidden_dim)
        v = self.value_up_proj(latent_kv)  # v: (batch, past_len+seq_len, hidden_dim)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # q: (batch, num_heads, seq_len, head_dim)
        k = k.view(batch, past_len + seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # k: (batch, num_heads, past_len + seq_len, head_dim)
        v = v.view(batch, past_len + seq_len, self.num_heads, self.head_dim). transpose(1, 2)  # v: (batch, num_heads, past_len + seq_len, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # scores: (batch, num_heads, seq_len, past_len + seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attention_weights = F.softmax(scores, dim = -1)  # attention_weights: (batch, num_heads, seq_len, past_len + seq_len)
        attention_output = torch.matmul(attention_weights, v)  # attention_output: (batch, num_heads, seq_len, head_dim)

        attention_output = attention_output.transpose(1, 2)  # attention_output: (batch, seq_len, num_heads, head_dim)
        attention_output = attention_output.reshape(batch, seq_len, self.hidden_dim)  # attention_output: (batch, seq_len, hidden_dim)

        # Output projection
        output = self.o_proj(attention_output)  # output: (batch, seq_len, hidden_dim)

        return output, new_latent_kv

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    num_heads = 2
    latent_dim = 4

    mla = MultiHeadLatentAttention(hidden_dim, num_heads, latent_dim)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).bool()  # causal mask
    output, new_latent_kv = mla(x, mask)

    logger.info(f"Output shape: {output.shape}, New latent KV shape: {new_latent_kv.shape}")