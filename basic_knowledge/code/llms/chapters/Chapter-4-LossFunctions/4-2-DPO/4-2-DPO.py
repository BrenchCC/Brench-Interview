import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta = 0.1
):
    """
    logps: (batch_size,) 每个值代表每条整段response的对数概率。
    """
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    ref_log_ratios = ref_chosen_logps - ref_rejected_logps

    # Compute loss
    logits = beta * (policy_log_ratios - ref_log_ratios)
    loss = - F.logsigmoid(logits)

    return loss.mean()

def get_sequence_logps(logits, labels, response_mask):
    """
    logits: (batch_size, seq_len, vocab_size)
    labels: (batch_size, seq_len)
    response_mask: (batch_size, seq_len)，1 表示 response token 参与计算
    """

    shift_logits = logits[:, :-1, :] # (batch_size, seq_len - 1, vocab_size)
    shift_labels = labels[:, 1:] # (batch_size, seq_len - 1)
    shift_mask = response_mask[:, 1:] # (batch_size, seq_len - 1)

    # 转换成对数softmax概率, log_probs: (batch, seq_len-1, vocab_size)
    log_probs = F.log_softmax(shift_logits, dim = -1)

    # 取出response token的对数概率, token_logps: (batch, seq_len-1, vocab_size)
    token_logps = log_probs.gather(
        dim = -1,
        index = shift_labels.unsqueeze(-1)  # (batch, seq_len-1, 1)，在vocab维度上gather
    ).squeeze(-1)

    # 过滤non-response token
    sequence_logps = (token_logps * shift_mask).sum(dim = -1)
    
    return sequence_logps


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    # 构造一个最小的 chosen/rejected 偏好数据批次 / Build a minimal preference batch.
    batch_size = 2
    seq_len = 6
    vocab_size = 32
    prompt_len = 3

    chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 只统计 response token 的 log probability，prompt 部分置零 / Mask prompt tokens.
    chosen_mask = torch.zeros(batch_size, seq_len)
    rejected_mask = torch.zeros(batch_size, seq_len)
    chosen_mask[:, prompt_len:] = 1
    rejected_mask[:, prompt_len:] = 1

    # 模拟 policy model 和 frozen reference model 的 logits / Simulate model logits.
    policy_chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
    policy_rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
    ref_chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
    ref_rejected_logits = torch.randn(batch_size, seq_len, vocab_size)

    # 获得policy上chosen和rejected的对数概率得分（每个response token的对数概率求和）
    policy_chosen_logps = get_sequence_logps(policy_chosen_logits, chosen_labels, chosen_mask)
    policy_rejected_logps = get_sequence_logps(policy_rejected_logits, rejected_labels, rejected_mask)
    
    # 获得ref上chosen和rejected的对数概率得分（每个response token的对数概率求和）
    with torch.no_grad():
        ref_chosen_logps = get_sequence_logps(ref_chosen_logits, chosen_labels, chosen_mask)
        ref_rejected_logps = get_sequence_logps(ref_rejected_logits, rejected_labels, rejected_mask)
    
    # 计算DPO损失
    loss = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta = 0.1
    )

    logger.info("Policy chosen log probabilities: %s", policy_chosen_logps)
    logger.info("Policy rejected log probabilities: %s", policy_rejected_logps)
    logger.info("DPO loss: %.6f", loss.item())
