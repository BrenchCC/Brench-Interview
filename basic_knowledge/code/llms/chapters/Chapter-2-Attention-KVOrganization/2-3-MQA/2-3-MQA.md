# 2.3 Multi-Query Attention (MQA)

多查询注意力机制。Q 使用 `num_heads` 个头，KV 使用 1 个头。

```python
import os
import sys
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_key_value_heads = 1  # K, V only have one head
      
        # K,V only have one head's parameters
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)
        self.key_proj = nn.Linear(hidden_dim, self.head_dim, bias = False)
        self.value_proj = nn.Linear(hidden_dim, self.head_dim, bias = False)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias = False)

        self.attention_dropout = nn.Dropout(0.1)

    def forward(self, x, mask = None):
        """
        x: (batch, seq_len, d_model)
        mask: (seq_len, seq_len), True=屏蔽, broadcast到batch和num_heads维度
        """
        # Get input x dimensions
        batch_size, seq_len, hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim, "Input hidden_dim must match module hidden_dim"

        # Compute Q, K, V
        q = self.query_proj(x)  # Q: (batch, seq_len, hidden_dim)
        k = self.key_proj(x)    # K: (batch, seq_len, head_dim)
        v = self.value_proj(x)  # V: (batch, seq_len, head_dim)

        # Split Q into multiple heads, K and V remain single head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # Q: (batch, num_heads, seq_len, head_dim)
        # k = k.unsqueeze(1)  # K: (batch, 1, seq_len, head_dim), insert a new dimension=1 on the specific position for the single head
        # v = v.unsqueeze(1)  # V: (batch, 1, seq_len, head_dim), insert a new dimension=1 on the specific position for the single head
        # Or use view and transpose to reshape K and V to have a single head dimension
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # K: (batch, 1, seq_len, head_dim)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # V: (batch, 1, seq_len, head_dim)

        # 
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # scores: (batch, num_heads, seq_len, seq_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))  # mask: (batch, 1, seq_len, seq_len)


        # Compute attention weights and output
        attention_weights = F.softmax(attention_scores, dim = -1)  # attn
        attention_weights = self.attention_dropout(attention_weights)  # Apply dropout to attention weights

        attention_output = torch.matmul(attention_weights, v)  # output: (batch, num_heads, seq_len, head_dim)

        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim)  # output: (batch, seq_len, hidden_dim)

        output = self.output_proj(attention_output)  # output: (batch, seq_len, hidden_dim)

        return output  # return: (batch, seq_len, hidden_dim)
  
if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    num_heads = 2

    x = torch.randn(batch_size, seq_len, hidden_dim)
    mqa = MultiQueryAttention(hidden_dim = hidden_dim, num_heads = num_heads)

    output = mqa(x)
    logger.info(f"Output shape: {output.shape}")  # Expected: (batch_size, seq_len, hidden_dim)

    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).bool()  # Lower triangular mask
    output_with_mask = mqa(x, mask = mask)
    logger.info(f"Output shape with mask: {output_with_mask.shape}")  # Expected: (batch_size, seq_len, hidden_dim)# Expected: (batch_size, seq_len, hidden_dim)
```

外部调用：

```python
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    num_heads = 2

    x = torch.randn(batch_size, seq_len, hidden_dim)
    mqa = MultiQueryAttention(hidden_dim = hidden_dim, num_heads = num_heads)

    output = mqa(x)
    logger.info(f"Output shape: {output.shape}")  # Expected: (batch_size, seq_len, hidden_dim)

    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal = 1).bool()  # Lower triangular mask
    output_with_mask = mqa(x, mask = mask)
    logger.info(f"Output shape with mask: {output_with_mask.shape}")  # Expected: (batch_size, seq_len, hidden_dim)
```

###
