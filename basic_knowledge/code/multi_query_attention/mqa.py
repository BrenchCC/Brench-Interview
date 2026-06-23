import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)  

class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head):
        super().__init__()
        assert hidden_dim % nums_head == 0, "hidden_dim must be divisible by nums_head"
        assert nums_head % nums_key_value_head == 0, "nums_head must be divisible by nums_key_value_head"

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head


        self.query_proj = nn.Linear(hidden_dim, nums_head * self.head_dim) # or self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.value_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, attention_mask = None):
        batch_size, seq_length, hidden_dim = x.size()

        # Project the input to queries, keys, and values
        query = self.query_proj(x)  # (batch_size, seq_length, nums_head * head_dim)
        key = self.key_proj(x)       # (batch_size, seq_length, nums_key_value_head * head_dim)
        value = self.value_proj(x)   # (batch_size, seq_length, nums_key_value_head * head_dim)

        # Reshape queries, keys, and values for multi-query attention
        # attenion_weights 目标是 (batch_size, nums_head, seq_length, seq_length)
        query = query.view(batch_size, seq_length, self.nums_head, self.head_dim).transpose(1, 2)  # (batch_size, nums_head, seq_length, head_dim)
        key = key.view(batch_size, seq_length, self.nums_key_value_head, self.head_dim).transpose(1, 2)  # (batch_size, nums_key_value_head, seq_length, head_dim)
        value = value.view(batch_size, seq_length, self.nums_key_value_head, self.head_dim).transpose(1, 2)  # (batch_size, nums_key_value_head, seq_length, head_dim)

        # 关注：nums_head and nums_key_value_head 关系
        # k,v repeat -> 广播操作
        key = key.repeat_interleave(self.nums_head // self.nums_key_value_head, dim = 1)
        value = value.repeat_interleave(self.nums_head // self.nums_key_value_head, dim = 1)

        
        attention_weights = torch.matmul(query, key.transpose(2,3)) / math.sqrt(self.head_dim) # or transpose(-2, -1)

        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(0, float("-1e20"))
        
        attention_weights = torch.softmax(attention_weights, dim = -1)

        output = attention_weights @ value

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_length, -1)
        output = self.out_proj(output)

        return output

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    attention_mask = None

    X = torch.randn(3, 2, 128)
    net = MultiQueryAttention(128, 8, 4)
    output = net(X)
    logger.info(f"The output shape is {output.shape}")

