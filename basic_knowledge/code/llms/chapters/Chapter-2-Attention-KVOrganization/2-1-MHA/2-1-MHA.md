# 2.1 Multi-Head Self-Attention (MHA)

多头自注意力机制。

```python
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
```

外部调用：

```python
x = torch.randn(2, 4, 8)
mha = MultiHeadAttention(d_model=8, num_heads=2)
mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
out = mha(x, mask=mask)
```

**pytorch 相关基础**：

- 如果 `assert` 希望给用户提示，可以写 `assert d_model % num_heads == 0, "d_model must be a multiples of num_heads"`。
- 如果传入的 `mask` 是 `(batch, seq_len, seq_len)` 的形状（通过 `if mask.dim() == 3` 来判断），需要先执行 `mask.unsqueeze(1)` 变为 `(batch, 1, seq_len, seq_len)`，否则广播机制对齐会出错。
