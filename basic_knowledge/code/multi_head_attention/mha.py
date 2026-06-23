import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head) -> None:
        super().__init__()
        self.nums_head = nums_head

        # hidden_dim: the dimension of the input and output of the multi-head attention
        # nums_head: the number of attention heads
        # head_dim: the dimension of each attention head, which is hidden_dim // nums_head
        # 一般来说，hidden_dim应该能够被nums_head整除，否则会导致head_dim不是整数，无法进行后续的计算
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // nums_head

        # 定义线性变换层，用于将输入的query、key、value映射到多头注意力的维度
        # 一般默认有bias，hidden_dim = nums_head * head_dim, 最终可以看成是n个矩阵的拼接
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # gpt2 和 bert 中都使用了dropout来防止过拟合，dropout的概率一般设置为0.1
        self.attention_dropout = nn.Dropout(0.1)

        # 定义输出的线性变换层，用于将多头注意力的输出映射回hidden_dim维度
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, X, attention_mask = None):
        # 在mask之前，先获取输入的batch_size、seq_len和hidden_dim，进行masked_fill
        # X: (batch_size, seq_len, hidden_dim)
        # attention_mask: (batch_size, seq_len), 其中1表示需要mask的部分，0表示不需要mask的部分
        batch_size, seq_len, hidden_dim = X.size()

        Q = self.query_proj(X)  # (batch_size, seq_len, hidden_dim)
        K = self.key_proj(X)    # (batch_size, seq_len, hidden_dim)
        V = self.value_proj(X)  # (batch_size, seq_len, hidden_dim)

        # 将Q、K、V拆分为多个头，shape变化，方便后续计算
        # Q: (batch_size, nums_head, seq_len, head_dim)
        # K: (batch_size, nums_head, seq_len, head_dim)
        # V: (batch_size, nums_head, seq_len, head_dim)
        q_state = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
        k_state = K.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
        v_state = V.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)

        # 计算注意力分数，使用缩放点积注意力机制，即将Q和K进行矩阵乘法，并除以sqrt(head_dim)而不是hidden_dim进行缩放，得到注意力分数
        # attention_scores: (batch_size, nums_head, seq_len, seq_len) 
        # torch.matmul(q_state, k_state.transpose(-2, -1)) / math.sqrt(self.head_dim) or using @ operator
        attention_weights = (
            q_state @ k_state.transpose(-2, -1) / math.sqrt(self.head_dim) 
        )
        logger.info(f"Attention weights type: {type(attention_weights)}")
        logger.info(f"Attention weights shape: {attention_weights.shape}")

        # 如果提供了attention_mask，则对注意力分数进行mask处理，将mask位置的注意力分数设置为一个很小的值，防止其对softmax归一化后的注意力权重产生影响
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask == 0, float("-1e20")
            )
        
        # 对注意力分数进行softmax归一化，得到注意力权重
        attention_weights = torch.softmax(attention_weights, dim = -1)
        logger.info(f"Attention weights after softmax: {(attention_weights)}")
        
        # 对注意力权重进行dropout处理，防止过拟合
        attention_weights = self.attention_dropout(attention_weights)

        output_state = attention_weights @ v_state  # (batch_size, nums_head, seq_len, head_dim)

        # shape变化，将多头注意力的输出拼接起来，得到最终的输出
        # (batch_size, nums_head, seq_len, head_dim) -> (batch_size, seq_len, hidden_dim)
        # contiguous()的作用是将tensor在内存中变为连续的，方便后续的计算, 一般用了permute/transpose之后都需要调用contiguous()，否则可能会报错
        # permute()的作用是改变tensor的维度顺序，transpose()的作用是交换tensor的两个维度，二者的区别在于permute()可以同时改变多个维度的顺序，而transpose()只能交换两个维度
        # 如果后续使用reshape，可以不调用contiguous()，因为reshape()会返回一个新的tensor，而不是在原来的tensor上进行操作，但是如果后续使用view()，则需要调用contiguous()，因为view()需要tensor在内存中是连续的，否则会报错
        output_state = output_state.permute(0, 2, 1, 3).contiguous()

        # 将多头注意力的输出拼接起来，得到最终的输出；-1表示自动计算该维度的大小，最终输出的shape为(batch_size, seq_len, hidden_dim)；
        # 如果不使用-1，而是使用self.hidden_dim，则需要保证self.hidden_dim = self.nums_head * self.head_dim，否则会报错
        output = output_state.view(batch_size, seq_len, -1)  # (batch_size, seq_len, hidden_dim)
        

        output = self.out_proj(output)  # (batch_size, seq_len, hidden_dim)

        return output
    
if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    attention_mask = None
    X = torch.randn(3, 2, 128)
    mha = MultiHeadAttention(hidden_dim = 128, nums_head = 8)
    output = mha(X, attention_mask)
    logger.info(f"Output type: {type(output)}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Output: {output}")
