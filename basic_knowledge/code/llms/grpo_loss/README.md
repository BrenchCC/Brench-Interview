# GRPO Loss 基本原理与代码维度分析

GRPO 的训练目标可以先写成下面这个形式。设同一个 prompt 采样出一组回答:

```math
\{o_i\}_{i = 1}^{G}
```

每条回答的 token 数为:

```math
|o_i|
```

当前策略、采样时的旧策略、参考策略分别为:

```math
\pi_\theta,\quad \pi_{\theta_{\text{old}}},\quad \pi_{\text{ref}}
```

代码里实现的 per-token surrogate loss 对应:

```math
J_{\text{GRPO}}(\theta) =
\frac{1}{G}
\sum_{i = 1}^{G}
\frac{1}{|o_i|}
\sum_{t = 1}^{|o_i|}
\left[
\min
\left(
r_{i,t}(\theta) A_i,
\operatorname{clip}(r_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon) A_i
\right)
- \beta D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})
\right]
```

其中:

```math
r_{i,t}(\theta)
=
\frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}
{\pi_{\theta_{\text{old}}}(o_{i,t} | q, o_{i,<t})}
=
\exp
\left(
\log \pi_\theta(o_{i,t}) - \log \pi_{\theta_{\text{old}}}(o_{i,t})
\right)
```

代码中的 KL 估计项为:

```math
D_{\text{KL}}(\pi_\theta || \pi_{\text{ref}})
\approx
\frac{\pi_{\text{ref}}(o_{i,t})}{\pi_\theta(o_{i,t})}
-
\log
\frac{\pi_{\text{ref}}(o_{i,t})}{\pi_\theta(o_{i,t})}
-
1
```

因为训练通常最小化 loss，代码最后对上面的 objective 取负号:

```math
\mathcal{L}_{\text{GRPO}}(\theta)
=
-J_{\text{GRPO}}(\theta)
```

本文对应 `grpo_loss.py`。我更建议把它当成面试用的最小实现:它保留了 GRPO 中最容易被追问的几块，分别是 group advantage、PPO-style ratio clipping、reference KL penalty、只在 response token 上计算 loss。代码不是完整训练框架，没有 reward model、采样器、优势归一化和分布式训练逻辑。

## 问题定义

当前示例里，batch 中的 3 条样本来自同一个 prompt 的 3 个回答:

```python
token_ids = torch.tensor(
    [
        [11, 12, 13, 14, 15],
        [11, 12, 13, 15, 16],
        [11, 12, 13, 16, 17],
    ]
)
```

前三个 token 是 prompt，后两个 token 是 response。因此:

| 符号 | 代码变量 | 示例值 | 含义 |
| --- | --- | ---: | --- |
| `B` | `bs` | 3 | group 内采样回答数，也可以理解为 batch size |
| `T` | `seq_len` | 5 | prompt token 与 response token 的总长度 |
| `V` | `vocab_size` | 32 | 词表大小 |
| `L_q` | `input_len` | 3 | prompt 长度 |
| `L_o` | `len_oi` | 2 | response 长度 |
| `G` | group size | 3 | 同一 prompt 下的回答数量 |

`advantage = torch.tensor([-1, 2, 1])` 是每条回答的组内优势。真实 GRPO 通常先对同一 prompt 的多个 reward 做归一化，例如:

```math
A_i =
\frac{r_i - \operatorname{mean}(\{r_1,\ldots,r_G\})}
{\operatorname{std}(\{r_1,\ldots,r_G\})}
```

这也是 GRPO 和传统 PPO 一个很大的差异:GRPO 不显式训练 value model，而是用 group 内相对分数估计优势。面试里可以直接说，GRPO 用多条采样回答的相对好坏替代 critic value，节省 value model 训练成本，但更依赖每个 prompt 下采样数量和 reward 质量。

## 从 logits 到 token log probability

示例先构造三套 logits:

```python
pi_logits = torch.randn(3, 5, 32)
pi_ref_logits = torch.randn(3, 5, 32)
pi_old_logits = torch.randn(3, 5, 32)
```

维度含义为:

```math
\text{logits} \in \mathbb{R}^{B \times T \times V}
```

当前策略、参考策略、旧策略的 shape 都是:

```python
(3, 5, 32)
```

随后计算 log probability:

```python
pi_logprob = F.log_softmax(pi_logits, dim = -1)
pi_ref_logprob = F.log_softmax(pi_ref_logits, dim = -1)
pi_old_logprob = F.log_softmax(pi_old_logits, dim = -1)
```

`F.log_softmax(x, dim = -1)` 沿最后一维做 softmax 后取 log。对 `(B, T, V)` 来说，就是对每个 batch、每个位置的 `V` 个候选 token 做归一化:

```math
\operatorname{log\_softmax}(z_j)
=
z_j - \log \sum_{k = 1}^{V} \exp(z_k)
```

shape 不变:

```python
(B, T, V) -> (B, T, V)
(3, 5, 32) -> (3, 5, 32)
```

此时还保留了所有词表 token 的 log probability。训练 loss 只需要真实生成 token 对应的那一个概率，所以代码用 `torch.gather` 取出目标 token:

```python
pi_logprob = torch.gather(
    pi_logprob,
    dim = -1,
    index = token_ids.unsqueeze(-1)
).squeeze(-1)
```

先看 `token_ids.unsqueeze(-1)`:

```python
token_ids:               (B, T)
token_ids.unsqueeze(-1): (B, T, 1)
```

示例中:

```python
(3, 5) -> (3, 5, 1)
```

`torch.gather(input, dim, index)` 会沿指定维度按 `index` 取值。这里 `dim = -1` 是词表维，所以每个位置只取出真实 token id 对应的 log probability:

```python
input:  (B, T, V)
index:  (B, T, 1)
output: (B, T, 1)
```

然后 `squeeze(-1)` 去掉最后那个长度为 1 的维度:

```python
(B, T, 1) -> (B, T)
(3, 5, 1) -> (3, 5)
```

这一步之后，`pi_logprob`、`pi_ref_logprob`、`pi_old_logprob` 都表示每个位置真实 token 的 log probability:

```math
\log \pi(o_t | q, o_{<t}) \in \mathbb{R}^{B \times T}
```

## KL 估计项

代码:

```python
def grpo_kl(pi_logprob, pi_ref_logprob):
    return pi_ref_logprob.exp() / pi_logprob.exp() - (pi_ref_logprob - pi_logprob) - 1
```

输入 shape:

```python
pi_logprob:     (B, T)
pi_ref_logprob: (B, T)
```

`Tensor.exp()` 对每个元素取指数。因为输入是 log probability，所以:

```python
pi_logprob.exp()     -> pi_prob
pi_ref_logprob.exp() -> pi_ref_prob
```

shape 不变:

```python
(B, T) -> (B, T)
```

第一项:

```python
pi_ref_logprob.exp() / pi_logprob.exp()
```

对应:

```math
\frac{\pi_{\text{ref}}(o_t)}{\pi_\theta(o_t)}
```

第二项:

```python
pi_ref_logprob - pi_logprob
```

对应:

```math
\log \pi_{\text{ref}}(o_t) - \log \pi_\theta(o_t)
=
\log
\frac{\pi_{\text{ref}}(o_t)}{\pi_\theta(o_t)}
```

所以整体为:

```math
x - \log x - 1,\quad x = \frac{\pi_{\text{ref}}(o_t)}{\pi_\theta(o_t)}
```

这个估计值非负，且当下面两个策略在该 token 上概率相同时取 0:

```math
\pi_\theta,\quad \pi_{\text{ref}}
```

代码返回:

```python
kl.shape = (B, T)
```

面试里被问到 KL 的作用，可以回答得具体一点:它不是为了让模型完全贴住 reference policy，而是限制 RL 更新不要把当前策略推离 SFT/reference 模型太远。否则 reward 只要有漏洞，模型很容易学到格式化投机、重复句式或其他 reward hacking 行为。

## `grpo_loss` 的计算流程

函数签名:

```python
def grpo_loss(
    pi_logprob,
    pi_old_logprob,
    pi_ref_logprob,
    advantage,
    input_len,
    len_oi
):
```

传入该函数时，三套 log probability 已经通过 `gather` 取到了真实 token 的概率:

| 变量 | shape | 示例 shape |
| --- | --- | --- |
| `pi_logprob` | `(B, T)` | `(3, 5)` |
| `pi_old_logprob` | `(B, T)` | `(3, 5)` |
| `pi_ref_logprob` | `(B, T)` | `(3, 5)` |
| `advantage` | `(B,)` | `(3,)` |
| `input_len` | scalar | `3` |
| `len_oi` | scalar | `2` |

### 基础超参数

代码:

```python
epsilon = 0.2
beta = 0.01
```

`epsilon` 控制 ratio clipping 的范围:

```math
r \in [1 - \epsilon, 1 + \epsilon]
```

`beta` 控制 KL penalty 的权重。`beta` 越大，当前策略越不容易偏离 reference policy；`beta` 越小，reward 对更新方向的影响越强。

### batch 与序列长度

代码:

```python
bs, seq_len = pi_logprob.shape
```

当前示例:

```python
bs = 3
seq_len = 5
```

这一步只是读 shape，不改变张量。

### response 长度张量

代码:

```python
len_oi = torch.tensor([len_oi] * bs, dtype = torch.long)
```

原始 `len_oi` 是一个标量 `2`。`[len_oi] * bs` 得到长度为 `B` 的列表:

```python
[2, 2, 2]
```

转成 tensor 后:

```python
len_oi.shape = (B,)
```

示例:

```python
(3,)
```

后面会用 `len_oi.unsqueeze(dim = 1)` 把它变成 `(B, 1)`，以便和 `(B, T)` 的 loss 做广播。这里有一个工程细节:如果 `pi_logprob` 在 GPU 上，这个新建 tensor 默认在 CPU，会触发 device mismatch。完整训练代码通常会写成:

```python
len_oi = torch.full(
    (bs,),
    len_oi,
    dtype = torch.long,
    device = pi_logprob.device
)
```

当前 README 只分析已有代码，不改实现。

### response mask

代码:

```python
mask = torch.zeros(bs, seq_len)
mask[:, input_len:] = 1
```

`torch.zeros(bs, seq_len)` 创建全 0 张量:

```python
mask.shape = (B, T)
```

示例:

```python
(3, 5)
```

`mask[:, input_len:] = 1` 表示所有 batch 行，从第 `input_len` 个位置开始置为 1。因为 `input_len = 3`，所以 mask 为:

```python
[
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
]
```

shape 仍是:

```python
(B, T)
```

它的含义很直接:prompt token 不参与 GRPO loss，只在 response token 上计算策略梯度和 KL penalty。真实训练里每条 response 长度可能不同，这时 mask 往往还要结合 padding mask；当前示例假设每条回答长度都等于 `len_oi = 2`。

### policy ratio

代码:

```python
ratio = torch.exp(pi_logprob - pi_old_logprob)
```

`pi_logprob - pi_old_logprob` 是逐元素相减:

```python
(B, T) - (B, T) -> (B, T)
```

`torch.exp(...)` 逐元素取指数，shape 不变:

```python
(B, T) -> (B, T)
```

它对应:

```math
r_{i,t}(\theta)
=
\exp
\left(
\log \pi_\theta(o_{i,t})
-
\log \pi_{\theta_{\text{old}}}(o_{i,t})
\right)
```

ratio 大于 1，说明当前策略比旧策略更倾向生成该 token；ratio 小于 1，则说明当前策略降低了该 token 的概率。

### ratio clipping

代码:

```python
ratio_clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
```

`torch.clamp(input, min, max)` 把每个元素截断到给定区间。这里区间是:

```python
[0.8, 1.2]
```

shape 不变:

```python
ratio:      (B, T)
ratio_clip: (B, T)
```

clipping 的目的和 PPO 一样:限制单次 policy update 的幅度。优势为正时，ratio 过大不再继续增加收益；优势为负时，ratio 过小也不会让目标函数无限受益。它让训练更稳，但也会带来保守更新。

### advantage 广播

代码:

```python
advantage = advantage.unsqueeze(dim = 1)
```

原始 `advantage` 每条回答一个值:

```python
advantage.shape = (B,)
```

`unsqueeze(dim = 1)` 在第 1 维插入长度为 1 的维度:

```python
(B,) -> (B, 1)
```

示例:

```python
[-1, 2, 1] -> [[-1], [2], [1]]
```

后续与 `ratio` 相乘时，PyTorch 会自动 broadcast:

```python
ratio:     (B, T)
advantage: (B, 1)
output:    (B, T)
```

这表示同一条回答中的所有 token 使用同一个 sequence-level advantage。GRPO 原始思想也是先对完整回答打分，再把这个相对优势分配到回答 token 上。

### clipped surrogate objective

代码:

```python
policy_gradient = torch.minimum(ratio * advantage, ratio_clip * advantage)
```

两个乘法都依赖广播:

```python
ratio * advantage:      (B, T) * (B, 1) -> (B, T)
ratio_clip * advantage: (B, T) * (B, 1) -> (B, T)
```

`torch.minimum(a, b)` 逐元素取较小值，shape 不变:

```python
(B, T)
```

这里容易被问到一个细节:为什么总是取 `minimum`，而不是根据 advantage 正负切换？PPO 的 clipped objective 写法本身就是:

```math
\min(r A, \operatorname{clip}(r, 1 - \epsilon, 1 + \epsilon) A)
```

当 advantage 为正时:

```math
A > 0
```

ratio 被限制在上界，避免过度提高好样本概率。当 advantage 为负时:

```math
A < 0
```

乘上负数会改变大小关系，`min` 会选择更保守的惩罚方向，避免模型过度降低坏样本概率。这个写法把正负 advantage 的情况合在一个公式里。

### KL penalty

代码:

```python
kl = grpo_kl(pi_logprob, pi_ref_logprob)
```

输入和输出:

```python
pi_logprob:     (B, T)
pi_ref_logprob: (B, T)
kl:             (B, T)
```

注意这里的 KL 是对已经 gather 出来的真实 token log probability 计算的 per-token penalty，而不是对整个 vocabulary 分布求完整 KL。完整 KL 需要保留 `(B, T, V)` 分布后在词表维求和，计算更贵。GRPO 实现里常见这种采样 token 上的估计项。

### 合并 policy gradient、KL 和 mask

代码:

```python
loss = (policy_gradient - beta * kl) * mask
```

逐项看 shape:

```python
policy_gradient: (B, T)
kl:              (B, T)
mask:            (B, T)
```

`beta * kl` 是标量乘矩阵:

```python
scalar * (B, T) -> (B, T)
```

`policy_gradient - beta * kl` 仍是:

```python
(B, T)
```

再乘 `mask` 后，prompt 位置被置 0，response 位置保留:

```python
(B, T) * (B, T) -> (B, T)
```

此时还没有取负号。它更接近要最大化的 objective 中每个 token 的贡献。

### 按 batch 和 response 长度归一化

代码:

```python
loss = (-1 / bs) * (1 / len_oi.unsqueeze(dim = 1)) * loss
```

先看 `len_oi.unsqueeze(dim = 1)`:

```python
len_oi:                   (B,)
len_oi.unsqueeze(dim = 1): (B, 1)
```

示例:

```python
[2, 2, 2] -> [[2], [2], [2]]
```

`1 / len_oi.unsqueeze(dim = 1)` 的 shape 是 `(B, 1)`。乘到 `(B, T)` 上时会 broadcast:

```python
(B, 1) * (B, T) -> (B, T)
```

`(-1 / bs)` 是标量，乘完 shape 仍是:

```python
(B, T)
```

这一步同时做了三件事:

1. 负号把最大化 objective 转成最小化 loss。
2. `1 / bs` 对 group/batch 内样本取平均。
3. `1 / len_oi` 对每条 response 的 token 数取平均。

当前示例中，每个 response 长度都是 2，因此每个 response token 的贡献会乘上:

```math
-\frac{1}{3} \times \frac{1}{2}
=
-\frac{1}{6}
```

### 求和得到标量 loss

代码:

```python
loss = loss.sum()
```

`Tensor.sum()` 默认对所有元素求和:

```python
(B, T) -> scalar
```

因为 prompt 位置已经被 mask 成 0，真正进入 loss 的只有 `B * L_o = 3 * 2 = 6` 个 response token。最终返回的是一个标量，可以直接用于:

```python
loss.backward()
```

## 当前示例的完整维度链路

把主函数中的关键张量串起来:

| 步骤 | 代码 | shape 变化 |
| --- | --- | --- |
| 构造 logits | `torch.randn(3, 5, 32)` | `(B, T, V)` |
| 归一化到 log prob | `F.log_softmax(..., dim = -1)` | `(B, T, V) -> (B, T, V)` |
| token id 扩维 | `token_ids.unsqueeze(-1)` | `(B, T) -> (B, T, 1)` |
| 取目标 token log prob | `torch.gather(..., dim = -1, index = ...)` | `(B, T, V) -> (B, T, 1)` |
| 去掉最后一维 | `.squeeze(-1)` | `(B, T, 1) -> (B, T)` |
| advantage 扩维 | `advantage.unsqueeze(dim = 1)` | `(B,) -> (B, 1)` |
| ratio | `torch.exp(pi_logprob - pi_old_logprob)` | `(B, T)` |
| clipping | `torch.clamp(ratio, 0.8, 1.2)` | `(B, T)` |
| surrogate | `torch.minimum(...)` | `(B, T)` |
| KL | `grpo_kl(...)` | `(B, T)` |
| response mask | `mask[:, input_len:] = 1` | `(B, T)` |
| 长度归一化 | `len_oi.unsqueeze(dim = 1)` | `(B,) -> (B, 1)` |
| 标量 loss | `loss.sum()` | `(B, T) -> scalar` |

## 面试问答要点

GRPO 和 PPO 的关系可以这样回答:GRPO 沿用了 PPO 的 ratio clipping 和 KL 约束，但去掉了 value model。它通过同一 prompt 下多条回答的 reward 相对值构造 advantage。这样实现更轻，但 reward 方差、采样数量和组内归一化会直接影响训练稳定性。

为什么只对 response token 算 loss？prompt 是条件输入，不是当前策略需要优化的输出。对 prompt token 算策略梯度没有意义，还会把监督信号污染到上下文部分。代码里的 `mask[:, input_len:] = 1` 就是这个边界。

为什么需要 old policy？ratio 的分母来自采样时的策略。PPO/GRPO 都假设样本由旧策略采样，再用当前策略做若干步更新。如果没有 old policy，ratio clipping 就失去了参照点。

为什么还要 reference policy？RL 优化只看 reward 时，模型可能找到奖励函数的漏洞。reference KL 相当于一个软约束，提醒当前策略不要离 SFT/reference 模型太远。

当前代码最需要注意的工程边界有两个。第一，`mask` 和 `len_oi` 默认建在 CPU，如果模型和 logprob 在 CUDA 上，需要显式指定 device。第二，示例假设所有 response 长度一致；真实 batch 中有 padding 时，`mask` 应该同时处理 prompt 区域和 padding 区域，否则 padding token 也可能进入损失。
