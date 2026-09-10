import os
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from ffn.SwiGLU import SwiGLU
from attention.mha import MultiHeadAttentionWithKVCache 
from normalization.RMSNorm import RMSNormalization


logger = logging.getLogger(__name__)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim):
        super().__init__()
        self.attention = MultiHeadAttentionWithKVCache(hidden_dim, num_heads)
        self.ffn = SwiGLU(hidden_dim, intermediate_dim)
        self.norm1 = RMSNormalization(hidden_dim)
        self.norm2 = RMSNormalization(hidden_dim)

    def forward(self, x, mask = None, past_kv = None):
        """
        x: (batch_size, seq_len, hidden_dim)
        mask: (seq_len, seq_len), causal mask / padding mask
        past_kv: None or tuple of (past_k, past_v)
        """
        # Pre-Norm + MHA + Residual Connection
        attn_out, new_kv = self.attention(self.norm1(x), mask = mask, past_kv = past_kv)
        x = x + attn_out
        # Pre-Norm + FFN + Residual Connection
        x = x + self.ffn(self.norm2(x))

        return x, new_kv

class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        num_heads,
        intermediate_dim,
        num_layers,
        max_seq_len
    ):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, num_heads, intermediate_dim) for _ in range(num_layers)
            ]
        )

        self.norm = RMSNormalization(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias = False)

    def forward(self, input_ids, mask = None, past_kv = None):
        """
        input_ids: (batch_size, seq_len), typically seq_len=1 for KV Cache
        mask: (seq_len, seq_len)
        past_kv: None or list of length num_layers
        past_kv[i]: (past_k, past_v) for the i-th layer
        past_k/past_v: (batch_size, num_heads, past_len, head_dim)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        past_len = past_kv[0][0].size(-2) if past_kv is not None else 0

        # Embedding and adding position encoding
        pos = torch.arange(past_len, past_len + seq_len, device = device) # pos: (seq_len, _)
        pos = pos.unsqueeze(0)  # (1, seq_len)
        x = self.word_embedding(input_ids) + self.position_embedding(pos)  # (batch_size, seq_len, hidden_dim)

        new_kv_list = []
        for i, layer in enumerate(self.layers):
            layer_past_kv = past_kv[i] if past_kv is not None else None
            x, new_kv = layer(x, mask = mask, past_kv = layer_past_kv)
            new_kv_list.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        return logits, new_kv_list

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )

    batch_size = 2
    vocab_size = 10000
    hidden_dim = 512
    num_heads = 8
    intermediate_dim = 2048
    num_layers = 12
    max_seq_len = 1024

    model = DecoderOnlyTransformer(
        vocab_size = vocab_size,
        hidden_dim = hidden_dim,
        num_heads = num_heads,
        intermediate_dim = intermediate_dim,
        num_layers = num_layers,
        max_seq_len = max_seq_len
    )

    past_kv = None

    for step in range(10):
        input_ids = torch.randint(0, vocab_size, (batch_size, 10))

        logits, past_kv = model(
            input_ids,
            mask = None,
            past_kv = past_kv
        )
        logger.info(f"logits: {logits.shape}")
        logger.info(f"past_kv: {len(past_kv)}")