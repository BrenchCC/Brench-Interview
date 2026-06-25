# RMSNorm 基础原理与代码维度分析

RMSNorm 的公式先放在开头:

$$
\operatorname{RMS}_{b,s}
=
\sqrt{
\frac{1}{D}
\sum_{i=1}^{D}
x_{b,s,i}^{2}
+
\epsilon
}
$$

$$
y_{b,s,i}
=
\gamma_i
\cdot
\frac{x_{b,s,i}}{\operatorname{RMS}_{b,s}}
$$

一句话概括:RMSNorm 不减均值,只用每个 token 在 hidden dimension 上的均方根缩放 hidden vector,再乘以可学习参数 `gamma`。

本文对应 `rmsnorm.py` 中的 `RMSNormalization` 实现。RMSNorm 可以看成 LayerNorm 的简化版本:它不减均值,只用 hidden dimension 上的均方根做缩放。面试里常见追问集中在三个点:它和 LayerNorm 的公式差异,为什么没有 `beta`,以及代码里的 `pow`、`mean`、`sqrt`、broadcasting 如何改变张量形状。

当前测试样例为:

```python
x = torch.randn(3, 2, 128)
_, _, feature_dim = x.shape
rn = RMSNormalization(feature_dim)

output = rn(x)
```

因此:

| 符号 | 代码变量 | 当前样例取值 | 含义 |
| --- | --- | ---: | --- |
| `B` | `x.shape[0]` | 3 | batch size |
| `S` | `x.shape[1]` | 2 | sequence length |
| `D` | `feature_dim` | 128 | hidden dimension |

输入 `x` 的形状是 `(B, S, D)`。RMSNorm 的输出仍然是 `(B, S, D)`。它不会改变 token 数量,也不会改变 hidden size。

## 问题定义

对 transformer 中常见的 hidden state:

$$
X \in \mathbb{R}^{B \times S \times D}
$$

RMSNorm 对每个样本、每个 token 的最后一维计算 root mean square。开头公式中的 `gamma` 是长度为 `D` 的可学习缩放参数。RMSNorm 通常没有 `beta`,也不做 `x - mean`。这正是它和 LayerNorm 的核心差异。

## RMSNorm 与 LayerNorm 的差异

LayerNorm 的标准形式是:

$$
y_i
=
\gamma_i
\cdot
\frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
+
\beta_i
$$

RMSNorm 的形式是:

$$
y_i
=
\gamma_i
\cdot
\frac{x_i}{\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2 + \epsilon}}
$$

两者都沿 hidden dimension 做归一化,统计量都来自单个 token 内部,因此都不依赖 batch size。区别在于 LayerNorm 会显式中心化,让归一化后的 hidden vector 均值接近 0;RMSNorm 只控制向量尺度,保留原始均值信息。

| 方法 | 是否减均值 | 分母 | 可学习参数 | 输出 shape |
| --- | --- | --- | --- | --- |
| LayerNorm | 是 | `sqrt(var + eps)` | `gamma`, `beta` | `(B, S, D)` |
| RMSNorm | 否 | `sqrt(mean(x^2) + eps)` | `gamma` | `(B, S, D)` |

面试中可以直接说:RMSNorm 不追求把 hidden state 中心化,它只把每个 token 的 hidden vector 缩放到相对稳定的 RMS 尺度。相比 LayerNorm,少了均值计算、中心化和偏置项,计算路径更短。

## 模块初始化

代码:

```python
class RMSNormalization(nn.Module):
    def __init__(self, feature_dim: int, eps: float = 1e-6):
        super().__init__()

        self.gemma = nn.Parameter(torch.ones(feature_dim))
        self.eps = eps
```

`feature_dim` 对应 hidden dimension。当前样例中:

```python
feature_dim = 128
```

所以可学习参数的形状为:

| 参数 | 初始化 | shape | 含义 |
| --- | --- | --- | --- |
| `self.gemma` | `torch.ones(feature_dim)` | `(128,)` | 缩放参数,标准记法通常写作 `gamma` |

`nn.Parameter` 会把 tensor 注册成模块参数。优化器拿到 `rn.parameters()` 后,会更新 `self.gemma`。

代码里变量名写成了 `gemma`。从数学含义看,它对应 RMSNorm 里的 `gamma`。文档和面试回答里建议称为缩放参数,这样不会被拼写细节带偏。

RMSNorm 没有 `beta`。原因不是不能加,而是经典 RMSNorm 设计只保留 rescaling invariance,不引入 re-centering。很多 LLM 实现也采用这种形式,例如 LLaMA 系列使用 RMSNorm 而不是标准 LayerNorm。

## 前向传播中的维度变化

### 平方

代码:

```python
x.pow(2)
```

`Tensor.pow(2)` 对张量逐元素平方,不改变 shape:

```python
x:         (B, S, D)
x.pow(2):  (B, S, D)
```

当前样例中:

```python
(3, 2, 128) -> (3, 2, 128)
```

这一步计算每个 hidden value 的平方,为后面的均方值做准备。

### 沿 hidden dimension 求均值

代码:

```python
x.pow(2).mean(-1, keepdim = True)
```

`mean(-1, keepdim = True)` 沿最后一维求均值。对 `(B, S, D)` 来说,就是对每个 token 的 `D` 个平方值求平均:

```python
x.pow(2):                         (B, S, D)
x.pow(2).mean(-1, keepdim=True):  (B, S, 1)
```

当前样例中:

```python
(3, 2, 128) -> (3, 2, 1)
```

`keepdim = True` 保留最后一维,方便后续和原始 `x` 做除法。若不保留维度,结果会是 `(B, S)`,后面需要手动 `unsqueeze(-1)` 才能和 `(B, S, D)` 对齐。

### 加 eps 与开方

代码:

```python
rms = torch.sqrt(self.eps + x.pow(2).mean(-1, keepdim = True))
```

`self.eps` 是标量,会 broadcast 到 `(B, S, 1)`:

```python
self.eps:                           scalar
x.pow(2).mean(...):                 (B, S, 1)
self.eps + x.pow(2).mean(...):      (B, S, 1)
```

`torch.sqrt(...)` 对每个位置逐元素开方,shape 不变:

```python
rms:  (B, S, 1)
```

当前样例中:

```python
(3, 2, 1) -> (3, 2, 1)
```

`eps` 的作用是数值稳定。若某个 token 的 hidden vector 接近全 0,均方值会很小,分母可能接近 0;加上 `eps` 后可以避免除法放大到不合理的范围。

### 除以 RMS

代码:

```python
x / rms
```

`x` 的 shape 是 `(B, S, D)`,`rms` 的 shape 是 `(B, S, 1)`。PyTorch 会沿最后一维 broadcast:

```python
x:          (B, S, D)
rms:        (B, S, 1)
x / rms:    (B, S, D)
```

当前样例中:

```python
(3, 2, 128) / (3, 2, 1) -> (3, 2, 128)
```

这里每个 token 只用一个 RMS 值缩放自己的 128 个 hidden channel。不同 token 的 RMS 值不同,不同 batch 样本之间也互不影响。

### 缩放参数广播

代码:

```python
output = self.gemma * x/rms
```

按 Python 运算符优先级,乘法和除法同级,从左到右计算。因此这行等价于:

```python
output = (self.gemma * x) / rms
```

由于 `self.gemma` 的 shape 是 `(D,)`,它会先 broadcast 到 `(B, S, D)`:

```python
self.gemma:       (D,)
x:                (B, S, D)
self.gemma * x:   (B, S, D)
```

然后再除以 `(B, S, 1)` 的 `rms`,输出 shape 仍是 `(B, S, D)`:

```python
(B, S, D) / (B, S, 1) -> (B, S, D)
```

当前样例中:

```python
(128,) * (3, 2, 128) / (3, 2, 1) -> (3, 2, 128)
```

从数学上看,由于都是逐元素乘除,该写法和下面形式等价:

```python
output = self.gemma * (x / rms)
```

为了阅读更清楚,工程代码里通常会加括号。当前实现的计算含义没有问题。

## 全流程维度表

以 `x = torch.randn(3, 2, 128)` 为例:

| 步骤 | 代码 | shape 变化 | 说明 |
| --- | --- | --- | --- |
| 输入 | `x` | `(3, 2, 128)` | batch 为 3,序列长度为 2,hidden size 为 128 |
| 平方 | `x.pow(2)` | `(3, 2, 128) -> (3, 2, 128)` | 对每个 hidden value 平方 |
| 均方 | `.mean(-1, keepdim = True)` | `(3, 2, 128) -> (3, 2, 1)` | 每个 token 内部沿 hidden dimension 求平均 |
| 加 eps | `self.eps + ...` | `(3, 2, 1) -> (3, 2, 1)` | 标量 `eps` broadcast 到均方张量 |
| RMS | `torch.sqrt(...)` | `(3, 2, 1) -> (3, 2, 1)` | 得到每个 token 一个 RMS 分母 |
| 缩放 | `self.gemma * x` | `(128,) * (3, 2, 128) -> (3, 2, 128)` | 每个 hidden channel 有独立缩放参数 |
| 归一化 | `/ rms` | `(3, 2, 128) / (3, 2, 1) -> (3, 2, 128)` | `rms` broadcast 到最后一维 |
| 输出 | `return output` | `(3, 2, 128)` | 输出形状与输入一致 |

## 相关 PyTorch 方法

`x.pow(2)` 对张量逐元素平方,不改变维度。也可以写成 `x ** 2`,当前代码使用 `pow` 更直观地表达了平方操作。

`mean(-1, keepdim = True)` 沿最后一维求均值。`-1` 表示最后一个维度,比写死维度编号更稳,只要 hidden dimension 始终在最后一维即可。

`torch.sqrt(...)` 对张量逐元素开方,不改变 shape。RMSNorm 中开方的对象是 `mean(x^2) + eps`,不是 variance。

`nn.Parameter(torch.ones(feature_dim))` 创建可学习缩放参数。初始化为 1 的含义是:训练开始时 RMSNorm 只做标准化缩放,不额外改变每个 hidden channel 的相对幅度。

PyTorch broadcasting 从尾维开始对齐。`(B, S, 1)` 可以和 `(B, S, D)` 做除法,因为最后一维的 `1` 能扩展成 `D`;`(D,)` 可以和 `(B, S, D)` 做乘法,因为它会对齐到最后一维。

## 面试回答要点

RMSNorm 的核心回答可以压缩成几句话:

1. RMSNorm 沿 hidden dimension 计算 `sqrt(mean(x^2) + eps)`,再用该值缩放当前 token 的 hidden vector。
2. 它不减均值,也通常没有 `beta`;相比 LayerNorm,计算路径更短。
3. 它的统计量来自单个 token 内部,不依赖 batch size,输入输出 shape 都是 `(B, S, D)`。
4. `gamma` 是长度为 `D` 的可学习缩放参数,通过 broadcasting 作用到每个 token。

一个常见追问是:RMSNorm 少了均值中心化,为什么仍然能用?可以这样回答:很多 transformer block 更关心 hidden state 的尺度稳定,而不是强制每个 token 的 hidden vector 均值为 0。残差连接会持续保留原始方向信息,RMSNorm 主要负责抑制激活尺度漂移。实际 LLM 中,RMSNorm 已经是很常见的选择,尤其在 LLaMA 这类 decoder-only 架构里。
