# 3.5 Transformer

定义 Transformer 单个模块以及整个 Decoder-only Transformer 架构。Transformer 原文架构如下。

![Transformer architecture](../../assets/transformer.png)

原文使用的是 Post-LN，下面代码实现使用现在更常用的 Pre-LN，梯度更稳定。另外，位置编码这里使用的是可学习编码，如果使用 RoPE，参考 3.3 小节直接修改调用的多头注意力类即可，不需要改动下面的代码（当然可学习编码要去掉）。

```python
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


```

定义 `top-k` 采样方法（在后续解码策略部分会再次提到该方法），之后进行外部调用：

```python
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
```

外部调用：

```python
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
```

**pytorch 相关基础**:

- `nn.Embedding(num_embeddings, embedding_dim)`，其中 `num_embeddings` 表示嵌入 token 的数量，`embedding_dim` 表示每个 token 映射到多少维度的向量。传入 Embedding 的张量可以是任意维度，但必须保证张量内每一个数值必须在 0 到 `num_embeddings-1`。
- `torch.Tensor.max(dim)` 返回一个元组：`(values, indices)`，里面保存了每个仅 `dim` 不同的每一组的最大值及其下标。如果存在多个最大值，返回的下标是第一次出现最大值的位置。
- 一般 `F.softmax` 本身内部已经做了数值稳定处理，但如果要显式地展示“防止溢出”技巧，可以手动先减最大值：

```python
def softmax(x, dim = -1):
    x = x - x.max(dim = dim, keepdim = True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim = dim, keepdim = True)
```

- `torch.multinomial()` 会在传入的参数中按数值比例采样，返回下标。要求输入只能是 1D 或 2D 的非负张量，`multinomial()` 会自动按权重比例采样（不一定求和要为1）。对于二维输入，默认对行采样，采样数量由 `num_samples` 决定。
- `torch.gather(input, dim, index)` 表示沿着 `dim` 维度，根据 `index` 里的下标，从 input 里取元素。注意 `input` 和 `index` 的维度数必须一样，输出 shape 和 `index.shape` 一样。
