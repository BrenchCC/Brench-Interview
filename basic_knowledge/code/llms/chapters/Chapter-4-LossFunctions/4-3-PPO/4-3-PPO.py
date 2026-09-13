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