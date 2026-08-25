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
        self.head_dim = self.hidden_dim // self.num_head


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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attention_weights = F.softmax(scores, dim = -1)
        attention_weights = self.attention_dropout(attention_weights)

        attention_output = torch.matmul(attention_weights, v)

        attention_output = attention_output.transpose(1, 2).contiguous()
        # attention_ourput = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim)

        output = self.output_proj(attention_output)

        return output

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
