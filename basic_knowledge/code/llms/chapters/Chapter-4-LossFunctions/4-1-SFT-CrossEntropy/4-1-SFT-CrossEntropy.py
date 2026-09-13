import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def cross_entropy(logits, labels):
    """
    logits: (batch_size * (seq_len - 1), vocab_size)
    labels: (batch_size * (seq_len - 1), )
    """

    # 防止softmax溢出, logits: (batch_size * (seq_len-1), vocab_size)
    logits = logits - torch.max(logits, dim = -1, keepdim = True).values

    # log_softmax: (batch_size * (seq_len - 1), vocab_size)
    exp_logits = torch.exp(logits)

    probs = exp_logits / exp_logits.sum(dim = -1, keepdim = True)
    log_probs = torch.log(probs)

    # 取对应标签id对应位置的负对数
    n = labels.size(0)
    loss = -log_probs[torch.arange(n, device = labels.device), labels]
    return loss.mean()

if __name__ == "__main__":
    # logits: (batch_size, seq_len, vocab_size)
    # labels: (batch_size, seq_len)
    
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    batch_size = 2
    seq_len = 5
    vocab_size = 16

    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # shift_logits: (batch, seq_len-1, vocab_size)，去掉最后一个token（结束符后面没有token了）
    shift_logits = logits[:, :-1, :].contiguous() 
    # shift_labels: (batch, seq_len-1)，去掉第一个token（起始不需要预测）
    shift_labels = labels[:, 1:].contiguous()

    loss = cross_entropy(
        shift_logits.view(-1, vocab_size), 
        shift_labels.view(-1)
    )
    torch_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1)
    )

    logger.info("Custom cross entropy: %.6f", loss.item())
    logger.info("PyTorch cross entropy: %.6f", torch_loss.item())
    logger.info("Close: %s", torch.allclose(loss, torch_loss, atol = 1e-6))

