import torch
import torch.nn.functional as F



def grpo_kl(pi_logprob, pi_ref_logprob):
    return pi_ref_logprob.exp() / pi_logprob.exp()- (pi_ref_logprob - pi_logprob) - 1

def grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, advantage, input_len, len_oi):
    epsilon = 0.2
    beta = 0.01

    bs, seq_len = pi_logprob.shape
    # skip计算采样的每条采样长度
    len_oi = torch.tensor([len_oi] * bs, dtype = torch.long)
    # 设定mask, 仅对response 为 1， 算loss
    mask = torch.zeros(bs, seq_len)
    mask[:, input_len:] = 1

    # GRPO loss
    ratio = torch.exp(pi_logprob - pi_old_logprob)
    ratio_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    advantage = advantage.unsqueeze(dim = 1) # [a, b ,c] -> [[a], [b], [c]]
    policy_gradient = torch.minimum(ratio * advantage , ratio_clip * advantage)
    kl = grpo_kl(pi_logprob, pi_ref_logprob)

    loss = (policy_gradient -  beta * kl) * mask
    loss = (-1 / bs ) * (1/len_oi.unsqueeze(dim = 1)) * loss  
    loss = loss.sum()

    return loss


if __name__ == "__main__":
    pi = torch.randn(3, 5) # batch, sequence
    pi_ref = torch.randn(3, 5) # batch, sequence
    pi_logprob = torch.nn.functional.log_softmax(pi, dim = 1)
    pi_ref_logprob = torch.nn.functional.log_softmax(pi_ref, dim = 1)
    print(grpo_kl(pi_logprob, pi_ref_logprob))

    # 输出分布
    pi_logits = torch.randn(3, 5, 32) # batch, seq_len, vocab_size
    pi_ref_logits = torch.randn(3, 5, 32)
    pi_old_logits = torch.randn(3, 5, 32)

    # 获取log prob
    pi_logprob = F.log_softmax(pi_logits, dim = -1)
    pi_ref_logprob = F.log_softmax(pi_ref_logits, dim = -1)
    pi_old_logprob = F.log_softmax(pi_old_logits, dim = -1)

    # group data
    token_ids = torch.tensor([[11, 12, 13, 14, 15], # 输入为11,12,13, 输出为:14, 15
                            [11, 12, 13, 15, 16],
                            [11, 12, 13, 16, 17],])

    # 获取policy
    pi_logprob = torch.gather(pi_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    pi_ref_logprob = torch.gather(pi_ref_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    pi_old_logprob = torch.gather(pi_old_logprob, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    loss = grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, torch.tensor([-1, 2, 1]), 3, 2)
    print(loss)