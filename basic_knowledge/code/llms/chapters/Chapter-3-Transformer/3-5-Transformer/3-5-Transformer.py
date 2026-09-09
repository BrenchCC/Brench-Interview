import os
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from ffn.SwiGLU import SwiGLU
from attention.mha import MultiHeadAttention
from normalization.RMSNorm import RMSNormalization

logger = logging.getLogger(__name__)

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim):
        super().__init__()

        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = SwiGLU(hidden_dim, intermediate_dim)

        self.norm1 = RMSNormalization(hidden_dim)
        self.norm2 = RMSNormalization(hidden_dim)

    def forward(self, x, mask = None):
        """
        x: (batch_size, seq_len, hidden_dim)
        mask: (seq_len, seq_len), causal mask / padding mask
        """
        # Pre-Norm + MHA + Residual Connection
        x = x + self.attention(self.norm1(x), mask = mask)

        # Pre-Norm + FFN + Residual Connection
        x = x + self.ffn(self.norm2(x))

        return x
    

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
        self.word_emb = nn.Embedding(vocab_size, hidden_dim)
        self.position_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim,
                    num_heads,
                    intermediate_dim
                ) for _ in range(num_layers)
            ]
        )

        self.norm = RMSNormalization(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias = False)

    def forward(self, input_ids, mask = None):
        """
        input_ids: (batch_size, seq_len)
        mask: (seq_len, seq_len)
        """
        # Get input_ids related information
        seq_len = input_ids.shape[-1]
        device = input_ids.device

        # Word embedding and add learnable position embedding
        pos = torch.arange(seq_len, device = device) # pos: (seq_len,)
        pos = pos.unsqueeze(0) # pos: (1, seq_len), broadcast到batch维度
        x = self.word_emb(input_ids) + self.position_emb(pos)

        # 注意力层处理，处理过程中始终保持x: (batch_size, seq_len, hidden_dim)
        for layer in self.layers:
            x = layer(x, mask = mask)

        x = self.norm(x)
        logits = self.lm_head(x) # LM_Head, logits: (batch_size, seq_len, vocab_size)

        return logits 

def top_k_sampling(logits, k = 50, temperature = 1.0):
    """
    logits: (batch_size, vocab_size)
    return: (batch_size, 1)
    """

    if temperature <= 0:
        return torch.argmax(logits, dim = -1, keepdim = True)

    # 温度+top-k采样, topk_logits/idx: (batch_size, k)
    logits = logits / temperature
    topk_logits, topk_idx = torch.topk(logits, k, dim = -1)

    # 只对top-k进行softmax归一化，并取下标映射回原始id
    probs = F.softmax(topk_logits, dim = -1)  # probs: (batch_size, k)
    sampled_idx = torch.multinomial(probs, num_samples = 1)  # sampled_idx: (batch_size, 1)
    next_token = torch.gather(topk_idx, dim = -1, index = sampled_idx)  # next_token: (batch_size, 1)

    return next_token

def softmax(x, dim = -1):
    x = x - x.max(dim = dim, keepdim = True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim = dim, keepdim = True)

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    batch_size = 2
    seq_len = 8
    vocab_size = 10000
    hidden_dim = 512
    num_heads = 8
    intermediate_dim = 2048
    num_layers = 6
    max_seq_len = 1024

    model = DecoderOnlyTransformer(
        vocab_size = vocab_size,
        hidden_dim = hidden_dim,
        num_heads = num_heads,
        intermediate_dim = intermediate_dim,
        num_layers = num_layers,
        max_seq_len = max_seq_len
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logger.info(f"Input ids: {input_ids}")

    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype = torch.bool),
        diagonal = 1
    )

    logits = model(input_ids, mask = mask)
    logger.info(f"Logits output: {logits}")
    next_token_id = top_k_sampling(logits[:, -1, :], k = 50, temperature = 0.8)  # next_token_id: (batch_size, 1)
    logger.info(f"Next token id shape: {next_token_id.shape}")
    logger.info(f"Next token id: {next_token_id}")

