import math
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class MultiHeadAttentionWithKVCache(nn.Module):
    def __init__(self, hidden_dim, nums_head) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        
        # hidden_dim: the dimension of the input and output of the multi-head attention
        # nums_head: the number of attention heads
        # head_dim: the dimension of each attention head, which is hidden_dim // nums_head
        # 一般来说，hidden_dim应该能够被nums_head整除，否则会导致head_dim不是整数，无法进行后续的计算
        self.head_dim = hidden_dim // nums_head


        # 定义线性变换层，用于将输入的query、key、value映射到多头注意力的维度
        # 一般默认有bias，hidden_dim = nums_head * head_dim, 最终可以看成是n个矩阵的拼接
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # gpt2 和 bert 中都使用了dropout来防止过拟合，dropout的概率一般设置为0.1
        self.attention_dropout = nn.Dropout(0.1)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 定义缓存，用于存储之前的key和value，以便在生成任务中进行高效的自回归推理
        self.register_buffer("key_cache", None, persistent = False)
        self.register_buffer("value_cache", None, persistent = False)
    
    def forward(self, x, attention_mask = None, past_kv_cache = None, store_kv_cache = False):
        # 在mask之前，先获取输入的batch_size、seq_len和hidden_dim，进行masked_fill
        # x: (batch_size, seq_len, hidden_dim)
        # attention_mask: (batch_size, seq_len), 其中1表示需要mask的部分，0表示不需要mask的部分
        batch_size, seq_len, hidden_dim = x.size()

        Q = self.query_proj(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key_proj(x)    # (batch_size, seq_len, hidden_dim)
        V = self.value_proj(x)  # (batch_size, seq_len, hidden_dim)

        # 将Q、K、V拆分为多个头，shape变化，方便后续计算
        # Q: (batch_size, nums_head, seq_len, head_dim)
        # K: (batch_size, nums_head, seq_len, head_dim)
        # V: (batch_size, nums_head, seq_len, head_dim)
        q_state = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3) # 或者transpose(2, 1) 交换维度2和维度1
        k_state = K.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3) # 或者transpose(2, 1) 交换维度2和维度1
        v_state = V.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3) # 或者transpose(2, 1) 交换维度2和维度1


        if past_kv_cache is not None:
            # 如果提供了past_kv_cache，则将其与当前的K和V进行拼接，形成新的K和V
            # 这样可以在生成任务中利用之前的上下文信息，提高生成的连贯性和准确性
            past_key, past_value = past_kv_cache
            k_state = torch.cat([past_key, k_state], dim = 2)  # (batch_size, nums_head, seq_len + past_seq_len, head_dim)
            v_state = torch.cat([past_value, v_state], dim = 2)  # (batch_size, nums_head, seq_len + past_seq_len, head_dim)
        
        # using torch.matmul(q_state, k_state.transpose(-2, -1)) / math.sqrt(self.head_dim) or using @ operator
        # 计算注意力分数，使用缩放点积注意力机制，即将Q和K进行矩阵乘法，并除以sqrt(head_dim)而不是hidden_dim进行缩放，得到注意力分数
        # 后续的计算中，注意力分数会经过softmax归一化，得到注意力权重，然后再与V进行矩阵乘法，得到最终的多头的单头注意力输出
        # 最后拼接所有头的输出，经过线性变换层映射回hidden_dim维度，得到最终的输出
        # attention_scores: (batch_size, nums_head, seq_len, seq_len) 

        attention_weights = torch.matmul(q_state, k_state.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, nums_head, seq_len, seq_len)

        logger.info(f"Attention weights type: {type(attention_weights)}")
        logger.info(f"Attention weights shape: {attention_weights.shape}")

        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(
                attention_mask == 0, float("-1e20")
            )
        attention_weights = torch.softmax(attention_weights, dim = -1)
        # logger.info(f"Attention weights after softmax: {(attention_weights)}")

        # 对注意力权重进行dropout处理，防止过拟合
        attention_weights = self.attention_dropout(attention_weights)

        # 计算多头注意力的输出，即将注意力权重与V进行矩阵乘法，得到每个头的输出
        output_state = torch.matmul(attention_weights, v_state)  # or using @ operator

        # shape变化，将多头注意力的输出拼接起来，得到最终的输出
        # (batch_size, nums_head, seq_len, head_dim) -> (batch_size, seq_len, hidden_dim)
        # contiguous()的作用是将tensor在内存中变为连续的，方便后续的计算, 一般用了permute/transpose之后都需要调用contiguous()，否则可能会报错
        # permute()的作用是改变tensor的维度顺序，transpose()的作用是交换tensor的两个维度，二者的区别在于permute()可以同时改变多个维度的顺序，而transpose()只能交换两个维度
        # 如果后续使用reshape，可以不调用contiguous()，因为reshape()会返回一个新的tensor，而不是在原来的tensor上进行操作，但是如果后续使用view()，则需要调用contiguous()，因为view()需要tensor在内存中是连续的，否则会报错
        output_state = output_state.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # or using output_state.view(batch_size, seq_len, hidden_dim)

        output = self.out_proj(output_state)  # (batch_size, seq_len, hidden_dim)

        if store_kv_cache:
            # 如果store_kv_cache为True，则将当前的K和V存储到缓存中，以便在生成任务中进行高效的自回归推理
            self.key_cache = k_state
            self.value_cache = v_state

        return output, (k_state, v_state)
    

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    attention_mask = None
    X = torch.randn(3, 2, 128)

    logger.info(f"---------- Testing MultiHeadAttentionWithKVCache ----------")
    logger.info(f"########## No past_kv_cache, store_kv_cache ##########")
    mha = MultiHeadAttentionWithKVCache(hidden_dim = 128, nums_head = 8)
    output, (k_state, v_state) = mha(X, attention_mask = attention_mask, past_kv_cache = None, store_kv_cache = True)
    logger.info(f"Output type: {type(output)}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Key state type: {type(k_state)}")
    logger.info(f"Key state shape: {k_state.shape}")
    logger.info(f"Value state type: {type(v_state)}")
    logger.info(f"Value state shape: {v_state.shape}")


    logger.info(f"########## With past_kv_cache, store_kv_cache ##########")
    k_state = torch.randn(3, 8, 2, 16)  # (batch_size, nums_head, past_seq_len, head_dim)
    v_state = torch.randn(3, 8, 2, 16)
    past_kv_cache = (k_state, v_state)
    output, (k_state, v_state) = mha(X, attention_mask = attention_mask, past_kv_cache = past_kv_cache, store_kv_cache = True)
    logger.info(f"Output type: {type(output)}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Key state type: {type(k_state)}")
    logger.info(f"Key state shape: {k_state.shape}")
    logger.info(f"Value state type: {type(v_state)}")
    logger.info(f"Value state shape: {v_state.shape}")


