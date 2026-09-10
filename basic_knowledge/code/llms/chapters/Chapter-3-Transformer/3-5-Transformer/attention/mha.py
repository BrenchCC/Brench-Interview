import os
import sys
import math
import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        # hidden_dim: the dimension of the input and output of the multi-head attention
        # nums_head: the number of attention heads
        # head_dim: the dimension of each attention head, which is hidden_dim // nums_head
        # 一般来说，hidden_dim应该能够被nums_head整除，否则会导致head_dim不是整数，无法进行后续的计算
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        assert hidden_dim % num_heads == 0, "d_model must be a multiples of num_heads"

        # 定义线性变换层，用于将输入的query、key、value映射到多头注意力的维度
        # 一般默认有bias，hidden_dim = nums_head * head_dim, 最终可以看成是n个矩阵的拼接
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)

        # 定义输出的线性变换层，用于将多头注意力的输出映射回hidden_dim维度
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)

        # gpt2 和 bert 中都使用了dropout来防止过拟合，dropout的概率一般设置为0.1
        self.attention_dropout = nn.Dropout(0.1)


    def forward(self, x, mask = None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len), True=屏蔽, broadcast到batch和num_heads维度
        """

        # Get x shape aka size
        batch_size, seq_len, _ = x.shape

        # Caculate QKV
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # QKV分头: (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力得分并添加掩码, scores:(batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # 计算注意力权重和输出
        attention_weights = F.softmax(scores, dim = -1)
        attention_weights = self.attention_dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, v)

        # 合并多头结果
        attention_output = attention_output.transpose(1, 2).contiguous() # out: (batch, seq_len, num_heads, head_dim)
        # attention_ourput = attention_output.permute(0, 2, 1, 3).contiguous() # out: (batch, seq_len, num_heads, head_dim)
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim) # (batch_size, seq_len, hidden_dim)

        output = self.output_proj(attention_output) # (batch_size, seq_len, hidden_dim)

        return output


class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias = False)
        self.k_proj = nn.Linear(d_model, d_model, bias = False)
        self.v_proj = nn.Linear(d_model, d_model, bias = False)
        self.o_proj = nn.Linear(d_model, d_model, bias = False)
        
    def forward(self, x, mask = None, past_kv = None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, past_len+seq_len), True=屏蔽, broadcast到batch和num_heads维度
        past_kv: tuple, (past_k, past_v)
        past_k/past_v: (batch, num_heads, past_len, head_dim)
        """
        # 获取x相关维度
        batch, seq_len, _ = x.shape
        
        # 计算QKV
        Q = self.q_proj(x)  # Q: (batch, seq_len, d_model)
        K = self.k_proj(x)  # K: (batch, seq_len, d_model)
        V = self.v_proj(x)  # V: (batch, seq_len, d_model)
        
        # QKV分头: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 使用KV Cache
        if past_kv is not None:
            past_k, past_v = past_kv
            # 新旧KV在seq_len维度拼接: (batch, num_heads, past_len+seq_len, head_dim)
            K = torch.cat([past_k, K], dim = -2)
            V = torch.cat([past_v, V], dim = -2)
        new_kv = (K, V)  # 更新KV Cache
        
        # 计算注意力得分并添加掩码, scores:(batch, num_heads, seq_len, past_len+seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # 计算注意力权重和输出
        attn_weights = F.softmax(scores, dim = -1)  # attn_weights: (batch, num_heads, seq_len, past_len+seq_len)
        out = torch.matmul(attn_weights, V)  # out: (batch, num_heads, seq_len, head_dim)
        
        # 合并多头结果
        out = out.transpose(1, 2)  # out: (batch, seq_len, num_heads, head_dim)
        out = out.contiguous().view(batch, seq_len, self.d_model)  # out: (batch, seq_len, d_model)
        
        # return: (batch, seq_len, d_model), KV Cache
        return self.o_proj(out), new_kv

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    x = torch.randn(2, 4, 8)
    mha = MultiHeadAttention(hidden_dim = 8, num_heads = 2)
    mask = torch.triu(torch.ones(4, 4, dtype = torch.bool), diagonal = 1)
    output = mha(x, mask = mask)
    
    logger.info(f"Output type: {type(output)}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Output: {output}")
