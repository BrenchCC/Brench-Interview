import math
import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionwithKVCache(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)

        self.attention_dropout = nn.Dropout(0.1)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)


    def forward(self, x, mask = None, past_kv = None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, past_len+seq_len), True=屏蔽, broadcast到batch和num_heads维度
        past_kv: tuple, (past_k, past_v)
        past_k/past_v: (batch, num_heads, past_len, head_dim)
        """

        # Get input shape
        batch_size, seq_len, _ = x.shape

        # Caculate Q K V
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        q = q.view(batch_size, seq_len, self.head_dim, self.num_heads)
        k = k.view(batch_size, seq_len, self.head_dim, self.num_heads)
        v = v.view(batch_size, seq_len, self.head_dim, self.num_heads)

        # Using KV cache
        if past_kv is not None:
            past_k, past_v = past_kv
            # 新旧KV在seq_len维度拼接: (batch, num_heads, past_len+seq_len, head_dim)
            K = torch.cat([past_k, k], dim = -2)
            V = torch.cat([past_v, v], dim = -2)

        new_kv = (k, v)  # 更新KV Cache

        # 计算注意力得分并添加掩码, scores:(batch, num_heads, seq_len, past_len+seq_len)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float('-inf'))


        # 计算注意力权重和输出
        attention_weights = F.softmax(attention_scores, dim = -1)  # attn_weights: (batch, num_heads, seq_len, past_len+seq_len)
        attention_weights = self.attention_dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, v)  # out: (batch, num_heads, seq_len, head_dim)

        # 合并多头结果
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim)

        output = self.output_proj(attention_output)

        return output, new_kv
    

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    batch_size = 2
    hidden_dim = 8
    num_heads = 2
    steps = 5

    mha = MultiHeadAttentionwithKVCache(hidden_dim, num_heads)
    past_kv = None

    for t in range(steps):
      # 自回归推理时，每一步只输入当前新 token
      x = torch.randn(batch_size, 1, hidden_dim)
      output, past_kv = mha(x, past_kv = past_kv)
    
      logger.info(f"step {t + 1}")
      logger.info(f"output:, {output.shape}")
      logger.info(f"cache_k:, {past_kv[0].shape}")
      logger.info(f"cache_v:, {past_kv[1].shape}")