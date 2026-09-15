import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def mask_mean(loss, mask = None):
    if mask is not None:
        return (loss * mask).sum() / mask.sum()
    else:
        return loss.mean()


def ppo_loss(
    new_logprobs,
    old_logprobs,
    advantages,
    values,
    returns,
    mask = None,
    ref_logprobs = None,
    clip_eps = 0.2,
    value_coef = 0.5,
    kl_coef = 0.1
):
    """
    new_logprobs: (batch_size, seq_len) 每时间步下新策略模型输出的对数概率
    old_logprobs: (batch_size, seq_len) 每时间步下旧策略模型输出的对数概率
    advantages: (batch_size, seq_len) 每时间步下优势估计
    values: (batch_size, seq_len) 每时间步下价值模型输出的预测价值
    returns: (batch_size, seq_len)  每时间步下的GAE回报
    mask: (batch_size, seq_len), 1 表示有效 token
    ref_logprobs: (batch_size, seq_len), 每时间步下参考模型输出的对数概率
    返回的是整个batch_size一起算出的loss
    """

    # 1. policy_loss
    ratio = torch.exp(new_logprobs - old_logprobs) #  新旧策略概率比r_t，去对数处理

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, min = 1.0 - clip_eps, max = 1.0 + clip_eps) * advantages

    policy_loss = -torch.min(unclipped, clipped)
    policy_loss = mask_mean(policy_loss, mask)

    # 2. value_loss
    value_loss = (values - returns) ** 2
    value_loss = mask_mean(value_loss, mask)

    # 3. kl_loss
    if ref_logprobs is not None:
        log_ratio = ref_logprobs - new_logprobs
        kl_loss = torch.exp(log_ratio) - 1.0 - log_ratio
        kl_loss = mask_mean(kl_loss, mask)
    else:
        kl_loss = 0.0

    total_loss = policy_loss + value_coef * value_loss + kl_coef * kl_loss

    return total_loss


def advantage_esitimate(
    rewards,
    values,
    dones,
    gamma = 0.99,
    lam = 0.95
):
    """
    rewards/values: (batch_size, seq_len)
    dones: (batch_size, seq_len), 1表示该轨迹结束，后续不再算values，0表示未结束
    gamma是折扣因子，lam是GAE平滑系数
    """
    # Get rewards information and initialize
    batch_size, seq_len = rewards.shape
    device = rewards.device
    advantages = torch.zeros_like(rewards) # (batch_size, seq_len)
    last_gae_lam = torch.zeros(batch_size, device = device) # (batch_size, _)

    for t in reversed(range(seq_len)):
        # 计算下一时刻起的values: V(s_{t+1})
        if t == seq_len - 1:
            next_values = torch.zeros(batch_size, device = device)
        else:
            next_values = values[:, t + 1]
        next_non_terminal = 1.0 - dones[:, t] # 是否达到该轨迹末端

        # TD ERROR: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_values * next_non_terminal - values[:, t]

        # 计算GAE
        last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam

        advantages[:, t] = last_gae_lam

    # 返回优势和折扣回报,advantages: (batch, seq_len), returns:(batch, seq_len)
    returns = advantages + values
    return advantages, returns

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    batch_size = 2
    seq_len = 6
    vocab_size = 32
    prompt_len = 3

    # 模拟已经采样出的 token 与三套策略 logits / Simulate sampled tokens and policy logits.
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    new_logits = torch.randn(batch_size, seq_len, vocab_size)
    old_logits = torch.randn(batch_size, seq_len, vocab_size)
    ref_logits = torch.randn(batch_size, seq_len, vocab_size)

    # 提取每个采样 token 的 log-prob / Gather log-prob for each sampled token.
    new_logprobs = torch.log_softmax(new_logits, dim = -1).gather(
        dim = -1,
        index = token_ids.unsqueeze(-1)
    ).squeeze(-1)
    old_logprobs = torch.log_softmax(old_logits, dim = -1).gather(
        dim = -1,
        index = token_ids.unsqueeze(-1)
    ).squeeze(-1)
    ref_logprobs = torch.log_softmax(ref_logits, dim = -1).gather(
        dim = -1,
        index = token_ids.unsqueeze(-1)
    ).squeeze(-1)

    # 仅优化 response token；每条轨迹在最后一个 token 终止 / Optimize response tokens only.
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, prompt_len:] = 1
    rewards = torch.randn(batch_size, seq_len) * response_mask
    values = torch.randn(batch_size, seq_len)
    dones = torch.zeros(batch_size, seq_len)
    dones[:, -1] = 1

    advantages, returns = advantage_estimate(rewards, values, dones)
    loss = ppo_loss(
        new_logprobs,
        old_logprobs,
        advantages,
        values,
        returns,
        mask = response_mask,
        ref_logprobs = ref_logprobs
    )

    logger.info("Advantages: %s", advantages)
    logger.info("PPO loss: %.6f", loss.item())
