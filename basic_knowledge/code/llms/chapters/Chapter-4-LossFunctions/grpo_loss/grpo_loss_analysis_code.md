# GRPO Loss Notebook：代码逐段展示

> 本页保留原 `grpo_loss_analysis.ipynb` 的代码、文本输出和图像输出。notebook 已删除；阅读本页不需要安装依赖或运行代码。

## 1. 导入依赖

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

本 notebook 的实际计算仅使用 `torch`；`nn` 与 `F` 在当前版本未参与后续单元。

## 2. KL 与组内优势函数

```python
def grpo_kl(pi_logprob, pi_ref_logprob):
    return (
        pi_ref_logprob.exp() / pi_logprob.exp()
        - (pi_ref_logprob - pi_logprob)
        - 1
    )


def grpo_advantage(rewards):
    epsilon = 0.00001
    advantage = (rewards - rewards.mean()) / (rewards.std() + epsilon)
    return advantage
```

`grpo_kl` 根据当前策略和参考策略的 log probability 计算逐元素 KL 近似项；当两者相同，该项为 0。

`grpo_advantage` 对同一组 reward 做标准化：高于组均值的 reward 产生正 advantage，低于组均值的 reward 产生负 advantage。所有 reward 相同时，advantage 全为 0。

## 3. 省略 clip/min 的最小 loss

```python
def minimal_grpo_loss(
    pi_logprob,
    pi_old_logprob,
    pi_ref_logprob,
    rewards,
    is_debug = True
):
    beta = 0.01
    kl = grpo_kl(pi_logprob, pi_ref_logprob)
    advantage = grpo_advantage(rewards)
    loss = -(
        torch.exp(pi_logprob - pi_old_logprob) * advantage
        - beta * kl
    )

    if is_debug:
        print('[Rewards]    :', rewards)
        print('[Advantage]  :', advantage)
        print('[Loss]       :', loss)

    return loss
```

这一单元使用的逐元素计算为：

```math
\ell_i = -\left(
\exp\left(\log\pi_i - \log\pi_{\mathrm{old},i}\right) A_i
- \beta\,\mathrm{KL}_i
\right)
```

它是观察符号与数值关系的简化版本：未实现 GRPO/PPO 正式目标中的 `min` 和 ratio clipping。这里的 `pi_logprob` 等参数在示例中均是标量，PyTorch 会将它们广播到 reward 组的每个元素。

## 4. 一条正奖励：逐元素 loss 有正也有负

```python
pi_logprob = torch.tensor(0.5).log()
pi_old_logprob = torch.tensor(0.5).log()
pi_ref_logprob = torch.tensor(0.6).log()
rewards_group = torch.tensor(
    [1, 0, 0, 0, 0, 0, 0, 0],
    dtype = torch.float32
)

loss = minimal_grpo_loss(
    pi_logprob,
    pi_old_logprob,
    pi_ref_logprob,
    rewards_group
)
loss.sum()
```

已保存输出：

```text
[Rewards]    : tensor([1., 0., 0., 0., 0., 0., 0., 0.])
[Advantage]  : tensor([ 2.4748, -0.3535, -0.3535, -0.3535,
                         -0.3535, -0.3535, -0.3535, -0.3535])
[Loss]       : tensor([-2.4746,  0.3537,  0.3537,  0.3537,
                         0.3537,  0.3537,  0.3537,  0.3537])
tensor(0.0014)
```

第一个样本的 advantage 为正，外层负号使其 loss 为负；其余样本的 advantage 为负，因此 loss 为正。`loss.sum()` 接近 0，是因为每个元素都使用同一个策略 ratio，标准化优势的正负部分相互抵消。

## 5. 放大当前策略与旧策略的 ratio

```python
pi_logprob = torch.tensor(0.1).log()
pi_old_logprob = torch.tensor(0.005).log()
pi_ref_logprob = torch.tensor(0.1001).log()
```

此处 `exp(pi_logprob - pi_old_logprob) = 20`，会放大 advantage 项。后续分别传入一条和两条正奖励：

```python
# one positive reward
rewards_group = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype = torch.float32)
loss = minimal_grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, rewards_group)
print(loss.sum(), '\n')

# two positive reward
rewards_group = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype = torch.float32)
loss = minimal_grpo_loss(pi_logprob, pi_old_logprob, pi_ref_logprob, rewards_group)
print(loss.sum())
```

| 正奖励数量 | 正优势位置的 loss | 负优势位置的 loss | 已保存 `sum(loss)` |
| ---: | ---: | ---: | ---: |
| 1 | `-49.4961` | `7.0709` | `-3.8147e-06` |
| 2 | `-32.4030` | `10.8010` | `3.8147e-06` |

该单元展示：较大的 ratio 会将正 advantage 对应的负 loss 放大。汇总值仍近似 0，原因仍是广播后的优势抵消，并不是每个元素的 loss 都为 0。

完整的已保存输出：

```text
[Rewards]    : tensor([1., 0., 0., 0., 0., 0., 0., 0.])
[Advantage]  : tensor([ 2.4748, -0.3535, -0.3535, -0.3535,
                         -0.3535, -0.3535, -0.3535, -0.3535])
[Loss]       : tensor([-49.4961,   7.0709,   7.0709,   7.0709,
                          7.0709,   7.0709,   7.0709,   7.0709])
tensor(-3.8147e-06)

[Rewards]    : tensor([1., 1., 0., 0., 0., 0., 0., 0.])
[Advantage]  : tensor([ 1.6202,  1.6202, -0.5401, -0.5401,
                         -0.5401, -0.5401, -0.5401, -0.5401])
[Loss]       : tensor([-32.4030, -32.4030,  10.8010,  10.8010,
                         10.8010,  10.8010,  10.8010,  10.8010])
tensor(3.8147e-06)
```

## 6. 只改变组内 reward 分布

固定策略概率：

```python
pi_logprob = torch.tensor(0.4).log()
pi_old_logprob = torch.tensor(0.3).log()
pi_ref_logprob = torch.tensor(0.401).log()
```

循环外调用使用四组 reward：

```python
[1, 0, 0, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 1, 0]
[1, 1, 1, 1, 1, 1, 1, 1]
```

| reward 为 1 的数量 | advantage 特征 | 已保存 `sum(loss)` |
| ---: | --- | ---: |
| 1 | 一个 `2.4748`，七个 `-0.3535` | `0.0` |
| 4 | 四个 `0.9354`，四个 `-0.9354` | `0.0` |
| 7 | 七个 `0.3535`，一个 `-2.4748` | `4.7684e-07` |
| 8 | 全部为 `0` | `2.4796e-07` |

最后一行中 reward 没有组内差异，标准化 advantage 为 0；故 policy 项没有信号，只保留极小的 KL 项。该段代码说明 GRPO 使用相对 reward，而不是简单使用 reward 的总和。

完整的已保存输出：

```text
[Rewards]    : tensor([1., 0., 0., 0., 0., 0., 0., 0.])
[Advantage]  : tensor([ 2.4748, -0.3535, -0.3535, -0.3535,
                         -0.3535, -0.3535, -0.3535, -0.3535])
[Loss]       : tensor([-3.2997,  0.4714,  0.4714,  0.4714,
                         0.4714,  0.4714,  0.4714,  0.4714])
result: 1.0 tensor(0.)

[Rewards]    : tensor([1., 1., 1., 1., 0., 0., 0., 0.])
[Advantage]  : tensor([ 0.9354,  0.9354,  0.9354,  0.9354,
                         -0.9354, -0.9354, -0.9354, -0.9354])
[Loss]       : tensor([-1.2472, -1.2472, -1.2472, -1.2472,
                         1.2472,  1.2472,  1.2472,  1.2472])
result: 4.0 tensor(0.)

[Rewards]    : tensor([1., 1., 1., 1., 1., 1., 1., 0.])
[Advantage]  : tensor([ 0.3535,  0.3535,  0.3535,  0.3535,
                          0.3535,  0.3535,  0.3535, -2.4748])
[Loss]       : tensor([-0.4714, -0.4714, -0.4714, -0.4714,
                         -0.4714, -0.4714, -0.4714,  3.2997])
result: 7.0 tensor(4.7684e-07)

[Rewards]    : tensor([1., 1., 1., 1., 1., 1., 1., 1.])
[Advantage]  : tensor([0., 0., 0., 0., 0., 0., 0., 0.])
[Loss]       : tensor([3.0994e-08, 3.0994e-08, 3.0994e-08, 3.0994e-08,
                        3.0994e-08, 3.0994e-08, 3.0994e-08, 3.0994e-08])
result: 8.0 tensor(2.4796e-07)
```

## 7. 固定 reward，逐步拉开与参考策略的距离

三个调用均使用：

```python
rewards_group = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype = torch.float32)
pi_ref_logprob = torch.tensor(0.401).log()
```

只把当前/旧策略同步从 `0.4` 改为 `0.3`、`0.1`：

| `pi_logprob` 与 `pi_old_logprob` 对应的概率 | 已保存 `sum(loss)` |
| ---: | ---: |
| 0.4 | `2.3842e-07` |
| 0.3 | `0.0037` |
| 0.1 | `0.1297` |

当当前策略远离 `0.401` 的参考策略，`grpo_kl` 增大，因而总体 loss 上升。这是本 notebook 中“loss rising”最直接的数值来源。

完整的已保存输出：

```text
[Rewards]    : tensor([1., 1., 0., 0., 0., 0., 0., 0.])
[Advantage]  : tensor([ 1.6202,  1.6202, -0.5401, -0.5401,
                         -0.5401, -0.5401, -0.5401, -0.5401])
[Loss]       : tensor([-1.6202, -1.6202,  0.5401,  0.5401,
                         0.5401,  0.5401,  0.5401,  0.5401])
result: 2.0 tensor(2.3842e-07)

[Rewards]    : tensor([1., 1., 0., 0., 0., 0., 0., 0.])
[Advantage]  : tensor([ 1.6202,  1.6202, -0.5401, -0.5401,
                         -0.5401, -0.5401, -0.5401, -0.5401])
[Loss]       : tensor([-1.6197, -1.6197,  0.5405,  0.5405,
                         0.5405,  0.5405,  0.5405,  0.5405])
result: 2.0 tensor(0.0037)

[Rewards]    : tensor([1., 1., 0., 0., 0., 0., 0., 0.])
[Advantage]  : tensor([ 1.6202,  1.6202, -0.5401, -0.5401,
                         -0.5401, -0.5401, -0.5401, -0.5401])
[Loss]       : tensor([-1.6039, -1.6039,  0.5563,  0.5563,
                         0.5563,  0.5563,  0.5563,  0.5563])
result: 2.0 tensor(0.1297)
```

## 8. reward 数量扫描与曲线

```python
pi_logprob = torch.tensor(0.1).log()
pi_old_logprob = torch.tensor(0.005).log()
pi_ref_logprob = torch.tensor(0.101).log()

nums = 128
rewards_group = torch.zeros(nums)
loss_list = []

for i in range(nums):
    rewards_group[i] = 1.0
    loss = minimal_grpo_loss(
        pi_logprob,
        pi_old_logprob,
        pi_ref_logprob,
        rewards_group,
        is_debug = False
    )
    loss_list.append(loss.sum().item())
```

该循环从全 0 reward 组开始，每次把一个位置设为 1，并记录整组 loss 的和。随后使用：

```python
plt.figure(figsize = (16, 6))
plt.plot(loss_list)
plt.title('grpo loss with rewards sum')
plt.xlabel('reward sum')
plt.ylabel('loss')
plt.grid()
plt.show()
```

生成曲线。由于使用同一标量 ratio 广播至 128 个样本，标准化 advantage 的和近似 0，图中的汇总值应主要表现为接近 0 的数值波动；它不是完整 GRPO trainer 的训练 loss 曲线。

原 notebook 保存的图像输出：

![随着正奖励数量增加，GRPO loss 汇总值在零附近波动](images/grpo_loss_with_reward_sum.png)

## 代码阅读要点

- 每个 `loss` 张量的元素对应 reward 组中的一个位置；仅看 `loss.sum()` 会掩盖正、负 loss 的分布。
- 本 notebook 的简化函数不含 `min`、clip、token mask 和 token/response 长度平均，不能直接替代训练实现。
- 实际训练应把平均 reward、KL、clip fraction 与平均 loss 一起监控，而非用单独的 loss 正负判断训练好坏。
