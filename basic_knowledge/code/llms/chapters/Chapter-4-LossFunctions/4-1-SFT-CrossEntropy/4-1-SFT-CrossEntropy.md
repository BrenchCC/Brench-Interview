# 4.1 SFT (Cross Entropy)

有监督微调 SFT 损失是 next-token prediction 的交叉熵损失：

```math
\mathcal{L}_{\text{SFT}}
=
-\frac{1}{N}
\sum_{t \in \text{response tokens}}
\log p_\theta(y_t \mid x, y_{\lt t})
```

其中：

- $x$：prompt
- $y_t$：第 $t$ 个目标 token
- $y_{\lt t}$：前面已经生成的目标 token

直接使用 torch 中的 `cross_entropy`：

```python
import torch.nn.functional as F

# logits: (batch, seq_len, vocab_size)
# labels: (batch, seq_len)

# shift_logits: (batch, seq_len-1, vocab_size)，去掉最后一个token（结束符后面没有token了）
shift_logits = logits[:, :-1, :].contiguous() 
# shift_labels: (batch, seq_len-1)，去掉第一个token（起始不需要预测）
shift_labels = labels[:, 1:].contiguous()

loss = F.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), ignore_index = -100)
```

自己实现 `cross_entropy`（一般不强制要求 `ignore_index` 参数）：

```python
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def cross_entropy(logits, labels):
    """
    logits: (batch_size * (seq_len - 1), vocab_size)
    labels: (batch_size * (seq_len - 1), )
    """

    # 防止softmax溢出, logits: (batch_size * (seq_len-1), vocab_size)
    logits = logits - torch.max(logits, dim = -1, keepdim = True).values

    # log_softmax: (batch_size * (seq_len - 1), vocab_size)
    exp_logits = torch.exp(logits)

    probs = exp_logits / exp_logits.sum(dim = -1, keepdim = True)
    log_probs = torch.log(probs)

    # 取对应标签id对应位置的负对数
    n = labels.size(0)
    loss = -log_probs[torch.arange(n, device = labels.device), labels]
    return loss.mean()

if __name__ == "__main__":
    # logits: (batch_size, seq_len, vocab_size)
    # labels: (batch_size, seq_len)
  
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    batch_size = 2
    seq_len = 5
    vocab_size = 16

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # shift_logits: (batch, seq_len-1, vocab_size)，去掉最后一个token（结束符后面没有token了）
    shift_logits = logits[:, :-1, :].contiguous() 
    # shift_labels: (batch, seq_len-1)，去掉第一个token（起始不需要预测）
    shift_labels = labels[:, 1:].contiguous()

    loss = cross_entropy(
        shift_logits.view(-1, vocab_size), 
        shift_labels.view(-1)
    )
    torch_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1)
    )

    logger.info("Custom cross entropy: %.6f", loss.item())
    logger.info("PyTorch cross entropy: %.6f", torch_loss.item())
    logger.info("Close: %s", torch.allclose(loss, torch_loss, atol = 1e-6))

```

**pytorch 基础**：

- `torch.nn.functional.cross_entropy()` 中 `ignore_index=-100` 表示 `label = -100` 的 token 不需要计算 loss，例如 padding 的地方或者 prompt 就可以设置其 `label = -100`。
