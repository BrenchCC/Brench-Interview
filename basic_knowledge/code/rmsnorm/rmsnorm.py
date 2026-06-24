import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class RMSNormalization(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-6): 
        super().__init__()

        self.gemma = nn.Parameter(torch.ones(feature_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(self.eps + x.pow(2).mean(-1, keepdim = True))
        output = self.gemma * x/rms
        return output

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    x = torch.randn(3, 2, 128)
    _, _, feature_dim = x.shape
    rn = RMSNormalization(feature_dim)

    output = rn(x)
    logger.info(output.shape)
    logger.info(output)