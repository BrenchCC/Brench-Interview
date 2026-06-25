# Self-Attention 基础原理与代码分析

本文对应 `self_attention.py` 中的单头 self-attention 实现。目标不是复述 transformer 论文，而是把面试里最容易被追问的几件事讲清楚：Q、K、V 的含义，注意力分数如何计算，mask 如何参与 softmax，以及每一步张量维度怎样变化。

## 1. 问题定义

输入 `x` 的形状为:

```python
x.shape = (batch_size, seq_length, dim)
```

在当前代码的测试样例中:

```python
X = torch.rand(3, 4, 2)
```

因此:

| 符号 | 代码变量 | 当前样例取值 | 含义 |
| --- | --- | --- | --- |
| `B` | `batch_size` | `3` | batch 中样本数量 |
| `L` | `seq_length` | `4` | 每个样本的 token 数 |
| `D` | `dim` | `2` | 每个 token 的 hidden size |

self-attention 的输出仍然保持 `(B, L, D)`。也就是说，每个 token 会读取同一个序列中其他 token 的信息，但不会改变序列长度和隐藏维度。

## 2. 公式

代码实现的是 scaled dot-product attention:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
S = \frac{QK^T}{\sqrt{D}}
$$

$$
A = \operatorname{softmax}(S)
$$

$$
O = AV
$$

其中 `S` 是 attention score，`A` 是归一化后的 attention weight，`O` 是每个 token 聚合上下文后的表示。除以 `sqrt(D)` 的目的很具体：当 `D` 增大时，`QK^T` 的点积方差会变大，softmax 更容易进入饱和区，梯度会变差。

## 3. Q、K、V 的直观含义

可以把每个 token 看成一次检索操作:

- `Q` 表示当前 token 想找什么信息。
- `K` 表示每个 token 能被匹配的索引。
- `V` 表示匹配成功后真正取出的内容。

一个常见面试追问是：为什么不用原始 `X` 直接做注意力？原因是模型需要在不同子空间里学习「匹配」和「取值」。`Q`、`K` 负责算相似度，`V` 负责提供被加权汇总的内容。三套线性投影允许模型把这两个角色分开。

## 4. 代码结构

核心模块如下:

```python
self.query_proj = nn.Linear(dim, dim)
self.key_proj = nn.Linear(dim, dim)
self.value_proj = nn.Linear(dim, dim)
self.attention_dropout = nn.Dropout(0.1)
self.output_proj = nn.Linear(dim, dim)
```

`nn.Linear(dim, dim)` 对最后一维做线性变换。输入是 `(B, L, D)` 时，PyTorch 会把前面的 `(B, L)` 当作 batch 维度保留，只变换最后一维:

```python
(B, L, D) -> (B, L, D)
```

当前实现是单头 attention，所以 `query_proj`、`key_proj`、`value_proj` 的输入输出维度都相同。如果扩展到 multi-head attention，通常会先投影到 `(B, L, num_heads * head_dim)`，再 reshape 成 `(B, num_heads, L, head_dim)`。

## 5. 前向传播中的维度变化

### 5.1 线性投影

代码:

```python
Q = self.query_proj(x)
K = self.key_proj(x)
V = self.value_proj(x)
```

维度:

| 变量 | 形状 | 当前样例 |
| --- | --- | --- |
| `x` | `(B, L, D)` | `(3, 4, 2)` |
| `Q` | `(B, L, D)` | `(3, 4, 2)` |
| `K` | `(B, L, D)` | `(3, 4, 2)` |
| `V` | `(B, L, D)` | `(3, 4, 2)` |

这里的 `Linear` 等价于对每个 token 独立做一次矩阵乘法。它不会让 token 之间发生信息交换，真正的 token 交互发生在后面的 `Q @ K.transpose(-2, -1)`。

### 5.2 计算 attention score

代码:

```python
attention_weights = Q @ K.transpose(-2, -1) / math.sqrt(self.dim)
```

先看 `K.transpose(-2, -1)`。`K` 的原始形状是 `(B, L, D)`，`transpose(-2, -1)` 交换倒数第二维和倒数第一维:

```python
K:                    (B, L, D)
K.transpose(-2, -1):  (B, D, L)
```

当前样例中:

```python
(3, 4, 2) -> (3, 2, 4)
```

然后用 `@` 做 batch matrix multiplication:

```python
Q @ K.transpose(-2, -1)
(B, L, D) @ (B, D, L) -> (B, L, L)
```

当前样例中:

```python
(3, 4, 2) @ (3, 2, 4) -> (3, 4, 4)
```

`(B, L, L)` 的含义是：对 batch 中每个样本，都得到一个 `L x L` 的注意力分数矩阵。第 `i` 行表示第 `i` 个 token 对所有 token 的关注程度，第 `j` 列表示被关注的第 `j` 个 token。

除以 `math.sqrt(self.dim)` 后形状不变:

```python
attention_weights.shape = (B, L, L)
```

变量名上有一个小细节：此时它还不是严格意义上的 weight，只是 score。经过 softmax 之后，每一行和为 1，才更适合称为 attention weight。

### 5.3 构造 mask

测试代码里先定义 padding mask:

```python
b = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ]
)
```

`b` 的形状是 `(B, L)`，当前为 `(3, 4)`。其中 `1` 表示该位置可见，`0` 表示该位置需要被屏蔽。

随后:

```python
mask = b.unsqueeze(dim = 1).repeat(1, 4, 1)
```

`unsqueeze(dim = 1)` 在第 1 维插入一个长度为 1 的维度:

```python
(B, L) -> (B, 1, L)
(3, 4) -> (3, 1, 4)
```

`repeat(1, 4, 1)` 将第 1 维复制 4 次:

```python
(B, 1, L) -> (B, L, L)
(3, 1, 4) -> (3, 4, 4)
```

该 mask 的每一行都相同。它表达的是「每个 query token 都不能看 padding token」。以第一个样本 `[1, 1, 1, 0]` 为例，扩展后每一行都是 `[1, 1, 1, 0]`，表示所有 query 位置都不能关注最后一个 key 位置。

如果做 causal attention，mask 会是下三角矩阵，含义会变成「当前位置不能看未来 token」。当前代码实现的是 padding mask，不是 causal mask。

### 5.4 应用 mask

代码:

```python
attention_weights = attention_weights.masked_fill(
    attention_mask == 0,
    float('-1e9')
)
```

`attention_mask == 0` 会得到一个 bool 张量，形状仍然是 `(B, L, L)`。`masked_fill` 会把 bool 张量中为 `True` 的位置替换成 `-1e9`。

为什么填 `-1e9` 而不是直接填 0？因为 mask 发生在 softmax 之前。若把被屏蔽位置的 score 设为 0，它在 softmax 后仍然可能得到非零概率。设为足够小的负数后，该位置的 `exp(score)` 接近 0，softmax 后权重也接近 0。

维度不变:

```python
(B, L, L) -> (B, L, L)
```

### 5.5 softmax 归一化

代码:

```python
attention_weights = torch.softmax(attention_weights, dim=-1)
```

`dim = -1` 表示沿最后一维做 softmax。对 `(B, L, L)` 来说，就是对每个 query token 的所有 key token 分数做归一化:

```python
attention_weights[b, i, :].sum() == 1
```

当前样例中，每个样本有 4 个 query token，所以每个样本会有 4 行概率分布。若 mask 生效，被屏蔽 key 位置的概率会接近 0。

维度仍然不变:

```python
(B, L, L) -> (B, L, L)
```

### 5.6 Dropout

代码:

```python
attention_weights = self.attention_dropout(attention_weights)
```

`nn.Dropout(0.1)` 在训练模式下会随机把一部分 attention weight 置 0，并对保留的部分做缩放，以保持期望不变。它常用于降低 attention 分布过度依赖少数位置的风险。

在评估模式下，也就是调用 `net.eval()` 后，Dropout 不会再随机丢弃元素。当前测试代码没有调用 `eval()`，所以每次运行时 attention weight 和 output 可能不同。

维度不变:

```python
(B, L, L) -> (B, L, L)
```

### 5.7 加权汇总 value

代码:

```python
attention_output = attention_weights @ V
```

矩阵乘法的维度是:

```python
(B, L, L) @ (B, L, D) -> (B, L, D)
```

当前样例中:

```python
(3, 4, 4) @ (3, 4, 2) -> (3, 4, 2)
```

这一步可以理解为：对每个 query token，用它那一行 attention weight 对所有 value 向量做加权平均。输出中第 `i` 个 token 的表示已经混入了它关注到的上下文信息。

### 5.8 输出投影

代码:

```python
output = self.output_proj(attention_output)
```

`output_proj` 仍然是 `nn.Linear(dim, dim)`，只作用于最后一维:

```python
(B, L, D) -> (B, L, D)
```

当前样例中:

```python
(3, 4, 2) -> (3, 4, 2)
```

在 multi-head attention 中，输出投影通常负责混合多个 head 拼接后的信息。当前是单头版本，输出投影更多是保持结构和真实 transformer block 对齐。

## 6. 完整维度流

| 步骤 | 表达式 | 形状变化 | 当前样例 |
| --- | --- | --- | --- |
| 输入 | `x` | `(B, L, D)` | `(3, 4, 2)` |
| Q 投影 | `query_proj(x)` | `(B, L, D)` | `(3, 4, 2)` |
| K 投影 | `key_proj(x)` | `(B, L, D)` | `(3, 4, 2)` |
| V 投影 | `value_proj(x)` | `(B, L, D)` | `(3, 4, 2)` |
| K 转置 | `K.transpose(-2, -1)` | `(B, D, L)` | `(3, 2, 4)` |
| attention score | `Q @ K.transpose(-2, -1)` | `(B, L, L)` | `(3, 4, 4)` |
| 缩放 | `/ sqrt(D)` | `(B, L, L)` | `(3, 4, 4)` |
| mask | `masked_fill(...)` | `(B, L, L)` | `(3, 4, 4)` |
| 归一化 | `softmax(dim = -1)` | `(B, L, L)` | `(3, 4, 4)` |
| attention dropout | `Dropout(0.1)` | `(B, L, L)` | `(3, 4, 4)` |
| 加权汇总 | `attention_weights @ V` | `(B, L, D)` | `(3, 4, 2)` |
| 输出投影 | `output_proj(...)` | `(B, L, D)` | `(3, 4, 2)` |

## 7. PyTorch 方法说明

| 方法 | 当前代码中的作用 | 面试中应说明的点 |
| --- | --- | --- |
| `nn.Linear(dim, dim)` | 对最后一维做线性投影 | 输入可以是高维张量，只要最后一维等于 `in_features` |
| `transpose(-2, -1)` | 交换 `K` 的序列维和特征维 | 使 `(B, L, D)` 变成 `(B, D, L)`，从而能和 `Q` 相乘 |
| `@` | batch matrix multiplication | 对三维张量会按 batch 维分别做矩阵乘法 |
| `math.sqrt(self.dim)` | 缩放 attention score | 防止点积过大导致 softmax 饱和 |
| `unsqueeze(dim = 1)` | 给 mask 增加 query 维 | `(B, L)` 变成 `(B, 1, L)` |
| `repeat(1, 4, 1)` | 将 mask 复制到每个 query 位置 | 当前写死了 `4`，更通用的写法是用 `seq_length` |
| `masked_fill(mask == 0, -1e9)` | 屏蔽不可见 key 位置 | 必须在 softmax 前做 |
| `torch.softmax(..., dim = -1)` | 对每个 query 的 key 分数归一化 | 最后一维代表被关注的 token |
| `nn.Dropout(0.1)` | 随机丢弃部分注意力权重 | 训练模式生效，评估模式关闭 |

## 8. 当前实现的边界

这份代码适合用来讲清楚 self-attention 的主干，但它不是完整的 transformer attention:

- 它是单头 attention，没有 `num_heads` 和 `head_dim` 的拆分。
- 它没有 residual connection、layer norm 和 feed-forward network。
- mask 构造里 `repeat(1, 4, 1)` 写死了序列长度，真实代码应改成 `repeat(1, seq_length, 1)`。
- `attention_mask` 当前需要和 score 一样广播到 `(B, L, L)`，如果传入 `(B, L)`，代码不会自动扩展。
- 使用 `float('-1e9')` 在 fp32 下通常可行；在混合精度训练中，真实工程里更常见的是根据 dtype 使用稳定的最小值或框架内置 mask 逻辑。

这些边界在面试里可以主动说出来。它能说明你不仅知道公式，也知道 demo 代码和生产实现之间差在哪里。

## 9. 面试回答模板

如果被问「self-attention 是怎么计算的」，可以按下面顺序回答:

1. 输入 `X` 的形状是 `(B, L, D)`，先通过三组线性层得到 `Q`、`K`、`V`，形状仍是 `(B, L, D)`。
2. 用 `Q @ K.transpose(-2, -1)` 得到 `(B, L, L)` 的 attention score，其中第 `i` 行表示第 `i` 个 token 对所有 token 的匹配分数。
3. score 除以 `sqrt(D)`，再在 softmax 前把 padding 或 future token 的位置填成很小的负数。
4. 沿最后一维做 softmax，得到每个 query 对所有 key 的概率分布。
5. 用这个概率分布乘以 `V`，得到 `(B, L, D)` 的上下文表示，最后再做一次输出投影。

一个短判断可以放在最后：self-attention 的本质是用内容相似度动态构造一个 `L x L` 的信息路由矩阵。相比 RNN 固定顺序传递信息，它让任意两个 token 可以在一层里直接交互；代价是 attention score 的空间和计算复杂度都和 `L^2` 相关。
