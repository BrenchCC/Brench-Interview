import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    def __init__(self, base_linear, r):
        """
        base_linear: nn.Linear(d_in, d_in)
        """
        super().__init__()

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        in_dim = base_linear.in_features
        out_dim = base_linear.out_features

        self.A = nn.Linear(in_dim, r, bias = False)
        self.B = nn.Linear(r, out_dim, bias = False)

        nn.init.zeros_(self.B.weight)  # A初始随机化，B初始为0

    def forward(self, x):
        return self.base(x) + self.B(self.A(x))


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    torch.manual_seed(42)
    input_dim = 8
    output_dim = 4
    rank = 2

    base_linear = nn.Linear(input_dim, output_dim)
    lora_linear = LoRALinear(base_linear, rank)
    x = torch.randn(2, 3, input_dim)

    # B 初始化为零，因此 LoRA 分支初始不会改变 base 输出 / Verify zero-initialized update.
    with torch.no_grad():
        base_output = base_linear(x)
        lora_output = lora_linear(x)

    logger.info(
        "Initial output matches frozen base: %s",
        torch.allclose(base_output, lora_output)
    )

    # 反向传播只更新 A、B；base 参数始终冻结 / Backpropagate through adapter parameters only.
    target = torch.randn_like(lora_output)
    loss = nn.functional.mse_loss(lora_linear(x), target)
    loss.backward()

    trainable_params = sum(
        parameter.numel()
        for parameter in lora_linear.parameters()
        if parameter.requires_grad
    )
    base_is_frozen = all(
        not parameter.requires_grad
        for parameter in lora_linear.base.parameters()
    )

    logger.info("LoRA loss: %.6f", loss.item())
    logger.info("Trainable adapter parameters: %d", trainable_params)
    logger.info("Base parameters frozen: %s", base_is_frozen)
