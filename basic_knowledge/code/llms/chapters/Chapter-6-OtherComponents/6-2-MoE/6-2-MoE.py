import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def moe_aux_loss(router_logits, topk_idx, num_experts):
    """
    router_logits: (num_tokens, num_experts)
    topk_idx: (num_tokens, top_k)
    """
    # 1. router 概率 pi
    router_probs = F.softmax(router_logits, dim = -1)  # router_probs: (num_tokens, num_experts)
    pi = router_probs.mean(dim = 0)  # pi: (num_experts,)

    # 2.token分配比例fi
    expert_mask = F.one_hot(topk_idx, num_classes = num_experts).float()
    fi = expert_mask.mean(dim = (0, 1)) # f1: (num_experts)

    aux_loss = num_experts * torch.sum(pi * fi)
    return aux_loss

class Expert(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim)
        )

    def forward(self, x):
        """
        x: (..., hidden_dim)
        return: (..., hidden_dim)
        """
        return self.net(x)

class MoE(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_experts, top_k = 3):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(hidden_dim, num_experts, bias = False)
        self.experts = nn.ModuleList(
            Expert(hidden_dim, intermediate_dim)
            for _ in range(num_experts)
        )

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        return: 线性层输出和负载均衡损失, out: (batch_size, seq_len, hidden_dim), aux_loss
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(batch_size * seq_len, hidden_dim)  # x_flat: (batch_size * seq_len, hidden_dim)

        # 计算路由到各个专家的分数, router_logits: (batch_size * seq_len, num_experts)
        router_logits = self.router(x_flat)

        # 每个token选取top-k专家并重新归一化, topk_logits/idx/probs: (batch_size * seq_len, top_k)
        topk_logits, topk_idx = torch.topk(router_logits, k = self.top_k, dim = -1)
        topk_probs = F.softmax(topk_logits, dim = -1)

        # 计算负载均衡损失，moe_aux_loss之后定义
        aux_loss = moe_aux_loss(
            router_logits = router_logits,
            topk_idx = topk_idx,
            num_experts = self.num_experts
        )

        # 初始化输出, out_flat: (batch_size * seq_len, hidden_dim)
        out_flat = torch.zeros_like(x_flat)


        for expert_id, expert in enumerate(self.experts):
            mask = topk_idx == expert_id
            if not mask.any():
                continue

            # 对于编号为token_idx的token，当前专家是它第which_k个top-k专家
            # token_idx和which_k是长度为selected_len的向量, selected_len为选中当前专家的token数
            token_idx, which_k = torch.where(mask)

            # expert processing: input: (selected_len, hidden_dim); output: (selected_len, hidden_dim)
            expert_input = x_flat[token_idx]
            expert_output = expert(expert_input)

            # 对于每个token, 将权重回传, weight: (selected_len, 1)
            weight = topk_probs[token_idx, which_k].unsqueeze(-1)
            out_flat.index_add_(dim = 0, index = token_idx, source = expert_output * weight)

        output = out_flat.view(batch_size, seq_len, hidden_dim)

        return output, aux_loss


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    torch.manual_seed(42)
    batch_size = 2
    seq_len = 4
    hidden_dim = 8
    intermediate_dim = 16
    num_experts = 4
    top_k = 2

    moe = MoE(hidden_dim, intermediate_dim, num_experts, top_k)
    x = torch.randn(batch_size, seq_len, hidden_dim)

    output, aux_loss = moe(x)
    loss = output.pow(2).mean() + 0.01 * aux_loss
    loss.backward()

    logger.info("MoE output shape: %s", output.shape)
    logger.info("Auxiliary load-balancing loss: %.6f", aux_loss.item())
    logger.info("Training loss: %.6f", loss.item())
    logger.info("Router received gradients: %s", moe.router.weight.grad is not None)
