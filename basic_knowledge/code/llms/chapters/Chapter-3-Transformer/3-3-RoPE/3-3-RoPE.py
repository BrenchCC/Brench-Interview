import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class RoPE(nn.Module):
    def __init__(self, hidden_dim, base = 10000):
        super().__init__()
        assert hidden_dim % 2 == 0
        self.hidden_dim = hidden_dim
        self.base = base

    def forward(self, x):
        """
        (x: batch_size, seq_len, hidden_dim)
        """
        _, seq_len, _ = x.shape
        device = x.device

        # Generate frequency, freq: (hidden_dim / 2,)   
        dim = torch.arange(0, self.hidden_dim, 2, device = device)  # dim: (hidden_dim / 2,)
        freq = self.base ** (-dim / self.hidden_dim)

        # Generate rotation angle, theta: (seq_len, hidden_dim / 2)
        pos = torch.arange(seq_len, device = device)  # pos: (seq_len,)
        # theta = pos[:, None] * freq[None, :]
        theta = torch.outer(pos, freq)
        logger.info(f"theta shape: {theta.shape}, theta: {theta}")

        # Calculate sine and cosine values, cos/sin: (seq_len, hidden_dim / 2)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        logger.info(f"cos shape: {cos.shape}, cos: {cos}")
        logger.info(f"sin shape: {sin.shape}, sin: {sin}")

        # x split into odd and even dimensions, x_odd/x_even: (batch_size, seq_len, hidden_dim / 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        logger.info(f"x_even shape: {x_even.shape}, x_even: {x_even}")
        logger.info(f"x_odd shape: {x_odd.shape}, x_odd: {x_odd}")

        # Calculate odd and even RoPE, out_odd, out_even: (batch_size, seq_len, hidden_dim / 2)
        out_even = cos * x_even - sin * x_odd
        out_odd = sin * x_even + cos * x_odd

        output = torch.zeros_like(x)
        output[..., 0::2] = out_even
        output[..., 1::2] = out_odd
        return output

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    # 测试RoPE
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    x = torch.randn(batch_size, seq_len, hidden_dim)
    rope = RoPE(hidden_dim)
    output = rope(x)
    logger.info(f"Input shape: {x.shape}, Output shape: {output.shape}")
    logger.info(f"Input: {x}")
    logger.info(f"Output: {output}")