import math
import logging

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim) -> None:
        super(SelfAttention, self).__init__()
        self.dim = dim

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        # 一般是 0.1 的dropout，在训练时随机丢弃一些连接，防止过拟合
        # 写作：config.attention_probs_dropout_prob
        # hidden_dropout_prob 一般也是0.1
        self.attention_dropout = nn.Dropout(0.1)

        # Multi-head attention 中的输出投影层产物，作用是将多头注意力的输出映射回原来的维度
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, x, attention_mask = None) :
        # attention_mask 是一个可选的张量，用于在计算注意力权重时屏蔽掉某些位置的影响，通常用于处理变长输入或掩盖未来信息
        # attention_mask shape = (batch_size, seq_length) 或 (batch_size, seq_length, seq_length)
        Q = self.query_proj(x)  
        K = self.key_proj(x)    
        V = self.value_proj(x)  

        attention_weights = Q @ K.transpose(-2, -1) / math.sqrt(self.dim)  # 计算注意力权重，除以 sqrt(dim) 是为了稳定训练过程中的梯度   
        batch_size, seq_length, dim = x.size()

        if attention_mask is not None:
            # 给 attention_weights 添加一个极小值（如 -1e9），使得被掩盖的位置在 softmax 后的权重接近于零
            attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-1e9'))

        attention_weights = torch.softmax(attention_weights, dim=-1)  # 对注意力权重进行 softmax 归一化
        logging.info("attention_weights shape: %s", attention_weights.shape)
        logging.info("attention_weights: %s", attention_weights)

        attention_weights = self.attention_dropout(attention_weights)  # 应用 dropout
        attention_output = attention_weights @ V  # 计算注意力输出
        output = self.output_proj(attention_output)  # 输出投影
        return output

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    X = torch.rand(3, 4, 2)  # (batch_size, seq_length, dim)
    b = torch.tensor(
        [
            [1, 1, 1, 0],  # 第一个样本的掩码，最后一个位置被掩盖
            [1, 1, 0, 0],  # 第二个样本的掩码，后两个位置被掩盖
            [1, 0, 0, 0]   # 第三个样本的掩码，后三个位置被掩盖
        ]
    )

    logging.info("Input X shape: %s", X.shape)

    mask = b.unsqueeze(dim = 1).repeat(1, 4, 1)

    net = SelfAttention(dim = 2)
    output = net(X, attention_mask = mask)
    logging.info("Output shape: %s", output.shape)
    logging.info("Output: %s", output)