import torch
import torch.nn.functional as F



def grpo_kl(pi_logprob, pi_ref_logprob):
    """Estimate the token-level KL divergence / 估计每个 token 的 KL 散度.

    Args:
        pi_logprob: 当前策略对已采样 token 给出的 log probability.
        pi_ref_logprob: Reference policy 对相同 token 给出的 log probability.

    Returns:
        A tensor with the same shape as the inputs, containing per-token KL estimates.
    """
    # Let x = log(pi_ref / pi). Then exp(x) - x - 1 is always non-negative.
    # 这种写法只需要采样 token 的 log-prob，可作为 KL(pi || pi_ref) 的无偏估计。
    return pi_ref_logprob.exp() / pi_logprob.exp()- (pi_ref_logprob - pi_logprob) - 1


def grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, advantage, input_len, len_oi):
    """Compute the GRPO objective / 计算一个 batch 的 GRPO loss.

    Args:
        pi_logprob: Current policy 对已采样 token 的 log-prob，shape 为 [batch, seq_len].
        pi_old_logprob: Rollout policy 对相同 token 的 log-prob，shape 为 [batch, seq_len].
        pi_ref_logprob: Reference policy 对相同 token 的 log-prob，shape 为 [batch, seq_len].
        advantage: 每条 response 的 group-relative advantage，shape 为 [batch].
        input_len: Prompt 占用的 token 数；该位置之前的 token 不参与 loss.
        len_oi: 每条 response 的有效 token 数，用于 length normalization.

    Returns:
        A scalar tensor representing the normalized GRPO loss.
    """
    # PPO-style clipping range / 限制新旧策略概率比的变化幅度。
    epsilon = 0.2
    # KL coefficient / 控制当前策略偏离 reference policy 的惩罚强度。
    beta = 0.01

    bs, seq_len = pi_logprob.shape
    # Demo assumes all responses have the same length / 此处假设 batch 内输出等长。
    # Real training data 通常会从 attention mask 逐条计算有效 response 长度。
    len_oi = torch.tensor([len_oi] * bs, dtype = torch.long)

    # Build a response-only mask: prompt = 0, generated response = 1.
    # 只优化模型生成部分，不对 prompt token 计算 policy loss。
    mask = torch.zeros(bs, seq_len)
    mask[:, input_len:] = 1

    # Importance sampling ratio: pi_theta(o_i) / pi_old(o_i).
    # 使用 log-prob 的差再取 exp，数值上也比先恢复两个概率再相除更稳定。
    ratio = torch.exp(pi_logprob - pi_old_logprob)

    # Clipped surrogate objective / 防止一次更新让 policy 变化过大。
    ratio_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

    # Broadcast one sequence-level advantage to every response token.
    # [a, b, c] -> [[a], [b], [c]], then broadcast along seq_len.
    advantage = advantage.unsqueeze(dim = 1)
    policy_gradient = torch.minimum(ratio * advantage , ratio_clip * advantage)

    # Penalize deviation from the fixed reference model / 抑制策略漂移。
    kl = grpo_kl(pi_logprob, pi_ref_logprob)

    # Maximize policy objective minus KL penalty; mask removes prompt positions.
    # PyTorch optimizer 默认最小化，因此下方乘以负号得到最终 loss。
    loss = (policy_gradient -  beta * kl) * mask

    # Average over samples and normalize each sample by its response length.
    # 这样较长的 response 不会仅因 token 更多而对梯度贡献更大。
    loss = (-1 / bs ) * (1/len_oi.unsqueeze(dim = 1)) * loss  
    loss = loss.sum()

    return loss


if __name__ == "__main__":
    # Part 1: verify the shape and values of the token-level KL estimator.
    pi = torch.randn(3, 5) # batch, sequence
    pi_ref = torch.randn(3, 5) # batch, sequence
    pi_logprob = torch.nn.functional.log_softmax(pi, dim = 1)
    pi_ref_logprob = torch.nn.functional.log_softmax(pi_ref, dim = 1)
    print(grpo_kl(pi_logprob, pi_ref_logprob))

    # Part 2: simulate policy outputs / 模拟三个模型输出的 vocabulary logits。
    pi_logits = torch.randn(3, 5, 32) # batch, seq_len, vocab_size
    pi_ref_logits = torch.randn(3, 5, 32)
    pi_old_logits = torch.randn(3, 5, 32)

    # Convert logits into log-probabilities over the vocabulary dimension.
    pi_logprob = F.log_softmax(pi_logits, dim = -1)
    pi_ref_logprob = F.log_softmax(pi_ref_logits, dim = -1)
    pi_old_logprob = F.log_softmax(pi_old_logits, dim = -1)

    # One prompt with three sampled responses / 同一 prompt 对应一组候选输出。
    token_ids = torch.tensor([[11, 12, 13, 14, 15], # 输入为11,12,13, 输出为:14, 15
                            [11, 12, 13, 15, 16],
                            [11, 12, 13, 16, 17],])

    # Select the log-prob assigned to each sampled token, reducing
    # [batch, seq_len, vocab_size] to [batch, seq_len].
    # Note: causal LM training 通常还需要将 logits 与 target token 错位对齐。
    pi_logprob = torch.gather(pi_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    pi_ref_logprob = torch.gather(pi_ref_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    pi_old_logprob = torch.gather(pi_old_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)

    # The three scalar advantages are shared by all response tokens in each row.
    loss = grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, torch.tensor([-1, 2, 1]), 3, 2)
    print(loss)
