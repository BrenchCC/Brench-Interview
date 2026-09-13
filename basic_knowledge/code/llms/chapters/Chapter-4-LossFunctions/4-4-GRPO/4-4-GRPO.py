import logging

import torch

logger = logging.getLogger(__name__)

def mask_mean(loss, mask = None):
    if mask is not None:
        return (loss * mask).sum() / mask.sum()
    else:
        return loss.mean()

def grpo_loss(
    new_logprobs,
    old_logprobs,
    rewards,
    group_size,
    mask = None,
    ref_logprobs = None,
    clip_eps = 0.2,
    kl_coef = 0.1,
    eps = 1e-8
):
    """
    new_logprobs: (batch_size, seq_len) 每时间步下新策略模型输出的对数概率
    old_logprobs: (batch_size, seq_len) 每时间步下旧策略模型输出的对数概率
    rewards: (batch_size,) 和PPO不同，PPO中returns是逐token的，GRPO中rewards对整个sequence计算
    group_size: GRPO中一组的大小
    mask: (batch_size, seq_len)
    ref_logprobs: (batch_size, seq_len) 每时间步下参考模型输出的对数概率
    clip_eps是clip中的eps，eps是计算组间相对优势时标准差用到的eps
    返回的是整个batch一起算出的loss
    """
    batch_size, _ = new_logprobs.shape
    assert batch_size % group_size == 0
    num_group = batch_size // group_size

    # 1. Compute relative advantage
    group_rewards = rewards.view(num_group, group_size) # group_rewards: (num_group, group_size)
    group_mean = group_rewards.mean(dim = -1, keepdim = True) # group_mean: (num_group, 1)
    group_std = group_rewards.std(dim = -1, keepdim = True, unbiased = False) # group_std: (num_group, 1)
    advantages = (group_rewards - group_mean) / (group_std + eps)
    advantages = advantages.view(batch_size, 1) # advantages: (batch_size, 1)

    # 2. policy loss
    ratio = torch.exp(new_logprobs - old_logprobs)
    unclipped = ratio * advantages

    clipped = torch.clamp(ratio, min = 1.0 - clip_eps, max = 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped)
    policy_loss = mask_mean(policy_loss, mask)

    # 3. kl_loss
    if ref_logprobs is not None:
        log_ratio = ref_logprobs - new_logprobs
        kl_loss = torch.exp(log_ratio) - 1.0 - log_ratio
        kl_loss = mask_mean(kl_loss, mask)
    else:
        kl_loss = 0.0

    # 4. total loss
    loss = policy_loss + kl_coef * kl_loss
    return loss


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    # 两组采样结果，每组包含三个 response / Two prompt groups with three responses each.
    batch_size = 6
    group_size = 3
    seq_len = 6
    vocab_size = 32
    prompt_len = 3

    # 模拟已经采样出的 token 序列 / Simulate sampled token sequences.
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 模拟新策略、旧策略和参考策略的词表 logits / Simulate policy logits.
    new_logits = torch.randn(batch_size, seq_len, vocab_size)
    old_logits = torch.randn(batch_size, seq_len, vocab_size)
    ref_logits = torch.randn(batch_size, seq_len, vocab_size)

    # 只保留每个采样 token 的 log-prob / Gather log-prob for sampled tokens.
    new_logprobs = torch.log_softmax(new_logits, dim = -1)
    old_logprobs = torch.log_softmax(old_logits, dim = -1)
    ref_logprobs = torch.log_softmax(ref_logits, dim = -1)

    new_logprobs = new_logprobs.gather(
        dim = -1,
        index = token_ids.unsqueeze(-1)
    ).squeeze(-1)
    old_logprobs = old_logprobs.gather(
        dim = -1,
        index = token_ids.unsqueeze(-1)
    ).squeeze(-1)
    ref_logprobs = ref_logprobs.gather(
        dim = -1,
        index = token_ids.unsqueeze(-1)
    ).squeeze(-1)

    # prompt token 不参与损失，response token 参与损失 / Mask prompt tokens.
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, prompt_len:] = 1

    # 每条 response 一个 sequence-level reward，同组内计算相对优势 / Sequence rewards.
    rewards = torch.tensor([-1.0, 0.5, 1.0, 0.8, 0.2, -0.6])

    loss = grpo_loss(
        new_logprobs,
        old_logprobs,
        rewards,
        group_size,
        mask = response_mask,
        ref_logprobs = ref_logprobs
    )

    logger.info("Group rewards: %s", rewards.view(-1, group_size))
    logger.info("GRPO loss: %.6f", loss.item())
