# 3.1 LayerNorm

层归一化。

```python
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class LayerNormalization(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super().__init__()
        self.gemma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True, unbiased = False)
        output = self.gemma * ((x - mean) / torch.sqrt(std + self.eps)) + self.beta
        return output

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    x = torch.randn(3, 2, 128)
    _, _, dim = x.shape
    ln = LayerNormalization(dim)
    output = ln(x)
    logger.info(x)
    logger.info(x.shape)
```

**pytorch 相关基础**

- 对张量某一维特征计算统计量聚合时，`keepdim=True` 保证聚合后该维度为1，并不会消失而减少维度。
