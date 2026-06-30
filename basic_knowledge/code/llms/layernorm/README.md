# LayerNorm 基础原理与代码维度分析

LayerNorm 的标准公式先放在开头:

```math
\mu_{b,s} = \frac{1}{D}\sum_{i=1}^{D}x_{b,s,i}
```

```math
\sigma^2_{b,s} = \frac{1}{D}\sum_{i=1}^{D}(x_{b,s,i} - \mu_{b,s})^2
```

```math
y_{b,s,i} =
\gamma_i
\cdot
\frac{x_{b,s,i} - \mu_{b,s}}{\sqrt{\sigma^2_{b,s} + \epsilon}}
+
\beta_i
```

一句话概括:LayerNorm 对每个 token 的 hidden dimension 单独计算均值和方差,再用可学习参数 `gamma`、`beta` 做缩放和平移。

本文对应 `layernorm.py` 中的 `LayerNormalization` 实现。重点放在面试中容易被追问的部分:LayerNorm 归一化的是哪一维,为什么它不依赖 batch size,可学习参数 `gamma` 和 `beta` 如何广播到输入张量,以及当前代码中每一步张量维度怎样变化。

当前测试样例为:

```python
x = torch.randn(3, 2, 128)
_, _, dim = x.shape
ln = LayerNormalization(dim)
output = ln(x)
```

因此:

| 符号 | 代码变量 | 当前样例取值 | 含义 |
| --- | --- | ---: | --- |
| `B` | `x.shape[0]` | 3 | batch size |
| `S` | `x.shape[1]` | 2 | sequence length |
| `D` | `dim` / `features` | 128 | hidden dimension |

输入 `x` 的形状是 `(B, S, D)`。LayerNorm 的输出仍然是 `(B, S, D)`。它不会改变 token 数量,也不会改变 hidden size,只是对每个 token 内部的 hidden dimension 做归一化。

## 问题定义

对 transformer 中常见的输入:

```math
X \in \mathbb{R}^{B \times S \times D}
```

LayerNorm 会对每个样本、每个 token 的最后一维单独计算均值和方差。也就是说,归一化单元是 `x[b, s, :]`,而不是整个 batch。开头公式中的 `gamma` 和 `beta` 都是长度为 `D` 的可学习参数。它们不随 batch 和 sequence 变化,只对应 hidden dimension 的每个通道。

## LayerNorm 与 BatchNorm 的差异

LayerNorm 和 BatchNorm 最容易混淆。面试里可以抓住归一化统计量的来源:

| 方法 | 统计量计算范围 | 对 batch size 的依赖 | 常见使用位置 |
| --- | --- | --- | --- |
| BatchNorm | 同一通道在 batch 维上的统计量 | 强依赖 batch size | CNN 更常见 |
| LayerNorm | 单个 token 内 hidden dimension 的统计量 | 不依赖 batch size | Transformer 更常见 |

对 NLP 和 LLM 来说,batch size 经常受序列长度、显存和 padding 影响。LayerNorm 只看单个 token 的 hidden vector,因此训练和推理时的行为更稳定,也更适合自回归生成。

## 模块初始化

代码:

```python
class LayerNormalization(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super().__init__()
        self.gemma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
```

这里 `features` 对应 hidden dimension。当前样例中:

```python
features = dim = 128
```

所以两个可学习参数的形状为:

| 参数 | 初始化 | shape | 含义 |
| --- | --- | --- | --- |
| `self.gemma` | `torch.ones(features)` | `(128,)` | 缩放参数,标准记法通常写作 `gamma` |
| `self.beta` | `torch.zeros(features)` | `(128,)` | 平移参数 |

`nn.Parameter` 会把普通 tensor 注册为模型参数。只要模块被优化器接管,`self.gemma` 和 `self.beta` 就会参与反向传播和参数更新。

代码里变量名写成了 `gemma`。从公式语义看,它对应的是 LayerNorm 里的 `gamma`。如果面试或后续代码中解释该变量,建议称为缩放参数,避免被拼写误差干扰理解。

## 前向传播中的维度变化

### 计算均值

代码:

```python
mean = x.mean(-1, keepdim = True)
```

`x.mean(dim = -1)` 表示沿最后一维求均值。对 `(B, S, D)` 来说,就是对每个 token 的 `D` 个 hidden value 求均值。

`keepdim = True` 会保留被归约的维度:

```python
x:     (B, S, D)
mean:  (B, S, 1)
```

当前样例中:

```python
(3, 2, 128) -> (3, 2, 1)
```

保留最后一维很重要。后面执行 `x - mean` 时,PyTorch 可以通过 broadcasting 自动把 `(B, S, 1)` 扩展到 `(B, S, D)`。

如果写成 `keepdim = False`,则 `mean` 的形状会变成 `(B, S)`。这时再和 `(B, S, D)` 相减,维度对齐会出问题,除非手动 `unsqueeze(-1)`。

### 计算标准差

代码:

```python
std = x.std(-1, keepdim =True, unbiased = False)
```

`torch.std(dim = -1)` 沿最后一维计算标准差,输出 shape 和 `mean` 一样:

```python
x:    (B, S, D)
std:  (B, S, 1)
```

当前样例中:

```python
(3, 2, 128) -> (3, 2, 1)
```

`unbiased = False` 表示使用有偏估计,分母是 `D`,不是 `D - 1`。LayerNorm 通常使用总体方差形式,因此这里选 `False` 是合理的。PyTorch 的 `nn.LayerNorm` 也使用 biased variance estimator。

这里有一个实现细节需要单独记住:`std` 已经是标准差,不是方差。标准 LayerNorm 的分母是 `sqrt(var + eps)`,等价于 `std` 版本里的 `std + eps`。如果已经调用 `x.std(...)`,通常不需要再对 `std` 做 `sqrt`。

### 中心化与归一化

代码:

```python
output = self.gemma * ((x - mean) / torch.sqrt(std + self.eps)) + self.beta
```

先看中心化:

```python
x:       (B, S, D)
mean:    (B, S, 1)
x-mean:  (B, S, D)
```

当前样例中:

```python
(3, 2, 128) - (3, 2, 1) -> (3, 2, 128)
```

这里发生了 broadcasting。`mean` 在最后一维只有 1 个值,会被自动扩展到 128 个 hidden channel,每个 token 内部的所有 hidden value 都减去同一个均值。

再看分母:

```python
std:                     (B, S, 1)
torch.sqrt(std + eps):   (B, S, 1)
```

除法同样通过 broadcasting 完成:

```python
(x - mean):                    (B, S, D)
torch.sqrt(std + self.eps):    (B, S, 1)
normalized:                    (B, S, D)
```

当前样例中:

```python
(3, 2, 128) / (3, 2, 1) -> (3, 2, 128)
```

注意,这行代码和标准 LayerNorm 公式存在差异。当前代码计算的是:

```math
\frac{x - \mu}{\sqrt{\operatorname{std}(x) + \epsilon}}
```

标准写法通常是下面两种之一:

```python
var = x.var(-1, keepdim = True, unbiased = False)
output = self.gemma * ((x - mean) / torch.sqrt(var + self.eps)) + self.beta
```

或:

```python
std = x.std(-1, keepdim = True, unbiased = False)
output = self.gemma * ((x - mean) / (std + self.eps)) + self.beta
```

两种写法的数学含义一致。当前实现多做了一次 `sqrt`,归一化强度会变弱:若标准差大于 1,分母从 `std` 变成 `sqrt(std)`,值更小;若标准差小于 1,分母从 `std` 变成 `sqrt(std)`,值更大。面试中建议按标准公式回答,代码实现则可以作为一次排查点记录下来。

### 缩放和平移

`self.gemma` 和 `self.beta` 的 shape 都是 `(D,)`:

```python
self.gemma:  (D,)
self.beta:   (D,)
```

乘法和平移时,PyTorch 会把它们 broadcast 到 `(B, S, D)`:

```python
self.gemma * normalized + self.beta
(D,) * (B, S, D) + (D,) -> (B, S, D)
```

当前样例中:

```python
(128,) * (3, 2, 128) + (128,) -> (3, 2, 128)
```

`gamma` 和 `beta` 的作用不是为了破坏归一化,而是把表达能力还给模型。纯归一化会强行把每个 token 的 hidden vector 拉到固定统计范围;加上可学习缩放和平移后,模型可以在需要时恢复某些通道的尺度和偏置。

## 全流程维度表

以 `x = torch.randn(3, 2, 128)` 为例:

| 步骤 | 代码 | shape 变化 | 说明 |
| --- | --- | --- | --- |
| 输入 | `x` | `(3, 2, 128)` | batch 为 3,序列长度为 2,hidden size 为 128 |
| 均值 | `x.mean(-1, keepdim = True)` | `(3, 2, 128) -> (3, 2, 1)` | 每个 token 内部沿 hidden dimension 求均值 |
| 标准差 | `x.std(-1, keepdim = True, unbiased = False)` | `(3, 2, 128) -> (3, 2, 1)` | 每个 token 内部沿 hidden dimension 求标准差 |
| 中心化 | `x - mean` | `(3, 2, 128) - (3, 2, 1) -> (3, 2, 128)` | `mean` broadcast 到最后一维 |
| 归一化 | `(x - mean) / denominator` | `(3, 2, 128) / (3, 2, 1) -> (3, 2, 128)` | 分母 broadcast 到最后一维 |
| 缩放 | `self.gemma * normalized` | `(128,) * (3, 2, 128) -> (3, 2, 128)` | 每个 hidden channel 有独立缩放参数 |
| 平移 | `+ self.beta` | `(3, 2, 128) + (128,) -> (3, 2, 128)` | 每个 hidden channel 有独立偏置参数 |
| 输出 | `return output` | `(3, 2, 128)` | 输出形状与输入一致 |

## 相关 PyTorch 方法

`x.mean(-1, keepdim = True)` 沿最后一维求平均。`-1` 表示最后一个维度,比写死 `2` 更通用:只要输入最后一维是 hidden dimension,无论前面有多少 batch-like 维度都能工作。

`x.std(-1, keepdim = True, unbiased = False)` 沿最后一维求标准差。`unbiased = False` 使用分母 `D`,适合 LayerNorm 这类归一化算子;`unbiased = True` 使用分母 `D - 1`,更偏统计估计语境。

`torch.sqrt(...)` 对张量逐元素开方,不改变 shape。标准 LayerNorm 里通常对 variance 加 `eps` 后开方;如果输入已经是 standard deviation,则直接加 `eps` 更符合公式。

`nn.Parameter(...)` 把 tensor 注册成模块参数。`model.parameters()` 能找到它,优化器也会更新它。普通 tensor 不会自动参与参数管理。

PyTorch broadcasting 会从尾维开始对齐维度。`(B, S, 1)` 可以和 `(B, S, D)` 运算,因为最后一维的 `1` 能扩展成 `D`;`(D,)` 也可以和 `(B, S, D)` 运算,因为它会对齐到最后一维。

## 面试回答要点

LayerNorm 的核心回答可以压缩成几句话:

1. LayerNorm 对每个 token 的 hidden dimension 单独归一化,输入输出 shape 都是 `(B, S, D)`。
2. 它的统计量来自单个样本内部,不依赖 batch size,因此适合 transformer 和自回归推理。
3. `gamma` 和 `beta` 是长度为 `D` 的可学习参数,通过 broadcasting 作用到每个 token。
4. 标准公式使用 `sqrt(var + eps)` 作为分母;如果代码先求 `std`,分母通常写成 `std + eps`。

一个常见追问是:LayerNorm 为什么放在 transformer block 里?回答时可以从训练稳定性说起。attention 和 MLP 会持续改变 hidden state 的分布,LayerNorm 把每个 token 的 hidden vector 拉回相对稳定的尺度,残差连接再负责保留原始信息通路。Pre-LN 结构中,LayerNorm 放在 attention/MLP 之前,深层网络的梯度通常更稳定;Post-LN 放在残差相加之后,早期 transformer 使用较多,但深层训练更难。
