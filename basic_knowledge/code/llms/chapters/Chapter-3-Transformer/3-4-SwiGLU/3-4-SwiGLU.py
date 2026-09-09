import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.W1 = nn.Linear(hidden_dim, intermediate_dim, bias = False)  # gate_proj
        self.W3 = nn.Linear(hidden_dim, intermediate_dim, bias = False)  # up_proj
        self.W2 = nn.Linear(intermediate_dim, hidden_dim, bias = False)  # down_proj


    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim)
        """
        # Compute gate and up branches  
        gate = F.silu(self.W1(x))  # gate: (batch, seq_len, intermediate_dim)
        value_up = self.W3(x)  # value_up: (batch, seq_len, intermediate_dim)

        output = gate * value_up  # Element-wise multiplication
        output = self.W2(output)  # down projection

        return output

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    # Example usage
    batch_size = 2
    seq_len = 5
    hidden_dim = 8
    intermediate_dim = 16

    model = SwiGLU(hidden_dim, intermediate_dim)
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
    output_tensor = model(input_tensor)

    logger.info(f"Input shape: {input_tensor.shape}")
    logger.info(f"Output shape: {output_tensor.shape}")