# Self-Attention 原理与 PyTorch 实现

本文结合 [`main.py`](./main.py) 中的单头 self-attention 实现，梳理计算公式、矩阵维度变化、mask 处理方式和相关 PyTorch API。重点不是背诵公式，而是能够在面试中解释：每一步为什么这样算，张量形状如何变化，以及该教学实现与标准 Transformer attention 的差异。

## 1. Self-Attention 解决的问题

给定一段长度为 \(L\) 的序列，每个 token 用一个 \(D\) 维向量表示。输入张量记作：

$$
X \in \mathbb{R}^{B \times L \times D}
$$

其中：

| 符号 | 含义 | `main.py` 中的取值 |
| --- | --- | --- |
| \(B\) | batch size，一批样本的数量 | 3 |
| \(L\) | sequence length，每个样本的 token 数量 | 4 |
| \(D\) | hidden dimension，每个 token 的特征维度 | 2 |

Self-attention 会为序列中的每个 token 计算一个新的表示。新表示由当前序列所有 token 的 Value 向量加权求和得到，权重取决于 Query 和 Key 的相似度。

对某个位置 \(i\) 而言，可以把计算理解为：

1. 用位置 \(i\) 的 Query 与所有位置的 Key 计算相似度。
2. 对相似度执行 softmax，得到位置 \(i\) 对所有位置的关注比例。
3. 按关注比例对所有 Value 加权求和，得到位置 \(i\) 的新表示。

Self-attention 中的 `self` 表示 Query、Key、Value 均由同一个输入 \(X\) 投影得到。若 Query 来自一个序列，而 Key 和 Value 来自另一个序列，则属于 cross-attention。

## 2. Scaled Dot-Product Attention

该实现对应单头 scaled dot-product attention：

$$
Q = XW_Q + b_Q
$$

$$
K = XW_K + b_K
$$

$$
V = XW_V + b_V
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

$$
Y = OW_O + b_O
$$

这里的 \(S\) 是 attention score，表示 token 两两之间的相关程度；\(A\) 是归一化后的 attention weight；\(Y\) 是最终输出。

### 为什么除以 \(\sqrt{D}\)

若 Query 和 Key 的各维元素彼此独立、均值为 0、方差为 1，则二者点积的方差会随维度 \(D\) 增长。维度较大时，点积容易得到绝对值很大的结果，使 softmax 进入饱和区间，梯度随之变小。

除以 \(\sqrt{D}\) 可以把点积结果控制在更稳定的数值范围内。当前代码是单头 attention，因此缩放因子使用 \(\sqrt{D}\)。在 multi-head attention 中，每个头通常使用：

$$
\sqrt{d_k}, \quad d_k = \frac{D}{H}
$$

其中 \(H\) 是 attention head 数量，\(d_k\) 是单个头的 Query 和 Key 维度。

## 3. 代码结构

`SelfAttention` 继承自 `nn.Module`，初始化阶段定义四个线性层和一个 dropout：

```python
class SelfAttention(nn.Module):
    def __init__(self, dim) -> None:
        super(SelfAttention, self).__init__()
        self.dim = dim

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(0.1)
        self.output_proj = nn.Linear(dim, dim)
```

三个输入投影层分别生成 \(Q\)、\(K\)、\(V\)。`output_proj` 对聚合后的结果再执行一次线性变换。当前实现只有一个 attention head，因此 `output_proj` 并不承担「拼接多个头」的工作；它仍然保留了标准 attention 模块中的输出投影形式。

### `nn.Linear(dim, dim)` 的参数与维度

`nn.Linear(in_features, out_features)` 对输入张量的最后一维执行线性变换，前面的 batch 和 sequence 维保持不变。

当 `dim = 2` 时，每个投影层的参数为：

| 参数 | 形状 |
| --- | --- |
| `weight` | `[2, 2]` |
| `bias` | `[2]` |

PyTorch 的 `nn.Linear` 可写作：

$$
Y = XW^T + b
$$

因此输入 `[3, 4, 2]` 经过 `nn.Linear(2, 2)` 后，输出仍为 `[3, 4, 2]`。线性层改变的是每个 token 的特征表示，不会混合不同 batch 或不同 token 的数据。

## 4. 完整维度变化

先使用通用维度表示整个计算流程：

| 计算步骤 | 运算 | 输入形状 | 输出形状 |
| --- | --- | --- | --- |
| 输入 | \(X\) | - | `[B, L, D]` |
| Query 投影 | `query_proj(x)` | `[B, L, D]` | `[B, L, D]` |
| Key 投影 | `key_proj(x)` | `[B, L, D]` | `[B, L, D]` |
| Value 投影 | `value_proj(x)` | `[B, L, D]` | `[B, L, D]` |
| Key 转置 | `K.transpose(-2, -1)` | `[B, L, D]` | `[B, D, L]` |
| 相关性计算 | `Q @ K.transpose(-2, -1)` | `[B, L, D] @ [B, D, L]` | `[B, L, L]` |
| 缩放 | `/ math.sqrt(dim)` | `[B, L, L]` | `[B, L, L]` |
| mask | `masked_fill(...)` | `[B, L, L]` | `[B, L, L]` |
| 权重归一化 | `softmax(..., dim = -1)` | `[B, L, L]` | `[B, L, L]` |
| dropout | `attention_dropout(...)` | `[B, L, L]` | `[B, L, L]` |
| Value 聚合 | `attention_weights @ V` | `[B, L, L] @ [B, L, D]` | `[B, L, D]` |
| 输出投影 | `output_proj(...)` | `[B, L, D]` | `[B, L, D]` |

在 `main.py` 的示例中：

$$
B = 3,\quad L = 4,\quad D = 2
$$

对应的具体形状为：

```text
X                         [3, 4, 2]
Q, K, V                   [3, 4, 2]
K.transpose(-2, -1)       [3, 2, 4]
Q @ K^T                   [3, 4, 4]
mask                      [3, 4, 4]
attention_weights         [3, 4, 4]
attention_weights @ V     [3, 4, 2]
output                    [3, 4, 2]
```

注意力计算会临时构造 `[B, L, L]` 的矩阵。序列长度翻倍时，该矩阵元素数量变为原来的四倍，因此标准 self-attention 对序列长度的时间复杂度和 attention matrix 空间复杂度均为：

$$
O(L^2)
$$

## 5. 逐段分析 `forward`

### 5.1 生成 Query、Key 和 Value

```python
Q = self.query_proj(x)
K = self.key_proj(x)
V = self.value_proj(x)
```

输入 `x` 的形状为 `[B, L, D]`。三个投影层分别学习不同参数，使同一个 token 在相关性查询、被查询索引和信息传递三个计算角色中拥有不同表示。

在当前示例中：

```text
x: [3, 4, 2]
Q: [3, 4, 2]
K: [3, 4, 2]
V: [3, 4, 2]
```

### 5.2 计算 token 两两相关性

```python
attention_weights = Q @ K.transpose(-2, -1) / math.sqrt(self.dim)
```

`K.transpose(-2, -1)` 交换最后两个维度：

```text
K                         [B, L, D]
K.transpose(-2, -1)       [B, D, L]
```

随后执行 batch matrix multiplication：

```text
[B, L, D] @ [B, D, L] -> [B, L, L]
```

得到的 `[B, L, L]` 矩阵可按行理解：

$$
S_{b,i,j}
$$

表示第 \(b\) 个样本中，第 \(i\) 个 token 的 Query 与第 \(j\) 个 token 的 Key 的点积结果。矩阵的倒数第二维 \(i\) 是「谁在查询」，最后一维 \(j\) 是「查询谁」。

对于长度为 4 的序列，每个样本会得到一个 `[4, 4]` 的 score matrix：

```text
              Key 位置
             0   1   2   3
Query 位置 0 [·   ·   ·   ·]
Query 位置 1 [·   ·   ·   ·]
Query 位置 2 [·   ·   ·   ·]
Query 位置 3 [·   ·   ·   ·]
```

### 5.3 构造并应用 padding mask

示例先定义有效 token 标记：

```python
b = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ]
)
```

`b` 的形状为 `[B, L] = [3, 4]`。其中 `1` 表示该 Key 位置有效，`0` 表示 padding 位置需要被屏蔽。

```python
mask = b.unsqueeze(dim = 1).repeat(1, 4, 1)
```

维度变化为：

```text
b                              [3, 4]
b.unsqueeze(dim = 1)           [3, 1, 4]
repeat(1, 4, 1)                [3, 4, 4]
```

`unsqueeze(dim = 1)` 在索引 1 的位置插入一个长度为 1 的维度。此时 `[B, 1, L]` 已经可以利用广播机制匹配 `[B, L, L]` 的 attention score；`repeat(1, 4, 1)` 则显式复制出 `[B, L, L]`，表示同一个样本中的每个 Query 都屏蔽相同的 padding Key。

随后执行：

```python
attention_weights = attention_weights.masked_fill(
    attention_mask == 0,
    float("-1e9")
)
```

`attention_mask == 0` 返回布尔张量。`masked_fill` 会把条件为 `True` 的位置替换为 `-1e9`。softmax 后，这些位置的权重接近 0：

$$
\frac{e^{-10^9}}{\sum_j e^{S_j}} \approx 0
$$

当前 `forward` 直接把 `attention_mask` 用于 `[B, L, L]` 的 score matrix。若传入原始 `[B, L]` mask，在一般的 \(B \ne L\) 情况下无法正确广播。因此，代码注释中提到的 `[B, L]` mask 需要先通过 `unsqueeze(1)` 转为 `[B, 1, L]`，再传入 `forward`。

`repeat` 会实际复制数据。这里只依赖广播时，可以保留 `[B, 1, L]`，减少额外内存占用：

```python
mask = b.unsqueeze(dim = 1)
```

### 5.4 softmax 归一化

```python
attention_weights = torch.softmax(attention_weights, dim = -1)
```

`dim = -1` 表示沿最后一维，也就是 Key 位置维度执行 softmax。对每个 Query 而言，其对所有可见 Key 的权重之和为 1：

$$
\sum_{j=1}^{L} A_{b,i,j} = 1
$$

不能在 Query 维度上执行 softmax。Attention 的目标是让每个 Query 决定如何分配对所有 Key 的关注比例。

### 5.5 对 attention weight 应用 dropout

```python
attention_weights = self.attention_dropout(attention_weights)
```

`nn.Dropout(0.1)` 在训练模式下随机把约 10% 的元素置为 0，并对保留元素按 \(1 / (1 - p)\) 缩放。该操作用于降低模型对特定注意力连接的依赖。

两个容易忽略的行为：

- 调用 `net.train()` 时启用 dropout；模块创建后默认处于训练模式。
- 调用 `net.eval()` 时关闭 dropout，推理结果不再因 dropout 随机变化。

因为 dropout 位于 softmax 之后，训练阶段每一行 attention weight 的实际和不一定等于 1。这是正常行为。

### 5.6 聚合 Value

```python
attention_output = attention_weights @ V
```

矩阵乘法维度为：

```text
[B, L, L] @ [B, L, D] -> [B, L, D]
```

对第 \(i\) 个 Query 位置：

$$
O_i = \sum_{j=1}^{L} A_{i,j}V_j
$$

因此，输出序列长度仍为 \(L\)，每个位置的特征维度仍为 \(D\)。变化发生在特征内容上：每个输出 token 已经融合了其可见范围内其他 token 的 Value 信息。

### 5.7 输出投影

```python
output = self.output_proj(attention_output)
```

`output_proj` 对最后一维执行线性变换：

```text
[B, L, D] -> [B, L, D]
```

标准 Transformer block 通常会在 attention 模块外继续执行 residual connection、dropout 和 LayerNorm。当前代码只实现 attention 主体，没有包含完整 Transformer block。

## 6. PyTorch 方法说明

### `torch.rand`

```python
X = torch.rand(3, 4, 2)
```

创建形状为 `[3, 4, 2]` 的浮点张量，元素从区间 `[0, 1)` 均匀采样。这里用于模拟一批已经完成 embedding 的 token 表示。

### `torch.tensor`

```python
b = torch.tensor([...])
```

根据给定数据创建张量。示例中的元素都是整数，因此 `b` 默认使用整型 dtype；执行 `b == 0` 后会得到布尔张量。

### `Tensor.size`

```python
batch_size, seq_length, dim = x.size()
```

返回张量各维度长度，效果与 `x.shape` 相近。当前 `forward` 解包得到这三个变量后没有继续使用，因此该行不影响计算结果。主程序构造 mask 时使用的 `repeat(1, 4, 1)` 也没有引用这里的 `seq_length`，其中的 `4` 是示例代码直接写入的序列长度。

### `Tensor.transpose`

```python
K.transpose(-2, -1)
```

交换指定的两个维度，不会颠倒其他维度。`-1` 表示最后一维，`-2` 表示倒数第二维。

### `@` 运算符

```python
Q @ K.transpose(-2, -1)
attention_weights @ V
```

`@` 对张量调用矩阵乘法语义。对于三维张量，PyTorch 将第一维视为 batch 维，并分别对每个 batch 执行矩阵乘法。

### `Tensor.unsqueeze`

```python
b.unsqueeze(dim = 1)
```

插入一个长度为 1 的新维度：

```text
[B, L] -> [B, 1, L]
```

长度为 1 的维度可以参与广播，常用于对齐 batch、head 或 sequence 维。

### `Tensor.repeat`

```python
b.unsqueeze(dim = 1).repeat(1, 4, 1)
```

按指定次数复制各维数据：

```text
[B, 1, L] -> [B, 4, L]
```

`repeat` 会分配并复制数据；若后续算子支持广播，通常无需显式复制。

### `Tensor.masked_fill`

```python
attention_weights.masked_fill(attention_mask == 0, float("-1e9"))
```

使用布尔 mask 替换指定位置。它返回新张量，不会原地修改原张量；原地版本为 `masked_fill_`。

### `torch.softmax`

```python
torch.softmax(attention_weights, dim = -1)
```

沿指定维度把任意实数转换为非负、和为 1 的权重。PyTorch 内部会采用数值稳定实现，避免直接计算较大指数导致溢出。

### `nn.Dropout`

```python
self.attention_dropout = nn.Dropout(0.1)
```

训练时随机丢弃元素，推理时保持输入不变。其行为由 module 的 `training` 状态控制，而不是由 `torch.no_grad()` 控制。

## 7. Mask 类型与形状

Attention 中常见的 mask 有 padding mask 和 causal mask。二者屏蔽的目标不同。

### Padding mask

Padding mask 屏蔽补齐位置，避免真实 token 关注无意义的 padding token。当前代码使用的就是 padding mask。

若原始 mask 形状为 `[B, L]`，推荐扩展为：

```python
padding_mask = padding_mask.unsqueeze(dim = 1)
```

得到 `[B, 1, L]`，再广播到每个 Query 位置。

### Causal mask

Decoder 的自回归生成要求位置 \(i\) 只能看到位置 \(0\) 到 \(i\)，不能看到未来 token。对应 mask 是下三角矩阵：

```text
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

其基础形状为 `[L, L]`，可以扩展为 `[1, L, L]` 或 `[B, L, L]`。实际 decoder attention 通常需要把 causal mask 与 padding mask 合并。

### 全行均被屏蔽的问题

若某个 Query 对应的一整行 Key 全部被屏蔽，softmax 将面对一行极小值。使用有限值 `-1e9` 时可能得到近似均匀分布；使用负无穷时则可能产生 `NaN`。工程实现需要保证有效 Query 至少能看到一个 Key，或在 softmax 后再次清理无效 Query 的输出。

## 8. 当前实现与标准 Multi-Head Attention 的差异

当前实现适合用来理解 attention 的主干计算，但它不是完整的 Transformer attention 模块。

| 对比项 | 当前实现 | 标准 Multi-Head Attention |
| --- | --- | --- |
| attention head | 单头 | 多头 |
| Q/K/V 形状 | `[B, L, D]` | `[B, H, L, d_k]` |
| 缩放因子 | `sqrt(D)` | `sqrt(d_k)` |
| mask 常见形状 | `[B, L, L]` 或可广播形状 | `[B, 1, 1, L]`、`[B, 1, L, L]` 等 |
| 输出投影 | 对单头输出做线性变换 | 拼接所有 head 后做线性变换 |
| residual / LayerNorm | 未实现 | 通常由 Transformer block 提供 |
| causal mask | 未内置 | decoder self-attention 通常需要 |

多头注意力会把特征维度拆成 \(H\) 个子空间：

$$
Q, K, V \in \mathbb{R}^{B \times H \times L \times d_k}
$$

每个 head 独立计算 `[L, L]` attention matrix，再把各 head 输出拼接回 `[B, L, D]`。不同 head 可以学习不同类型的关系，但不能保证每个 head 都会形成可直接解释的语言模式。

## 9. 面试问答要点

### Self-attention 为什么能够建模长距离依赖

任意两个 token 都可以通过一次 \(QK^T\) 计算直接建立联系，其路径长度不随 token 间距离增长。相比循环神经网络逐步传递状态，self-attention 更容易在并行计算中处理远距离关系。代价是 attention matrix 的大小随 \(L^2\) 增长。

### 为什么 Q、K、V 要使用不同投影

同一个 token 在注意力中承担三种角色：发起查询、提供匹配索引、传递实际信息。独立投影让模型分别学习这三种表示。若直接令 \(Q = K = V = X\)，表达能力会受限。

### softmax 为什么沿最后一维计算

score matrix 的形状是 `[B, L_query, L_key]`。最后一维对应一个 Query 能看到的所有 Key。沿该维执行 softmax 后，每个 Query 都得到一组对 Key 的权重分布。

### 为什么输出形状仍是 `[B, L, D]`

每个 Query 位置都会生成一个加权 Value 向量。Value 的特征维度为 \(D\)，Query 的数量为 \(L\)，因此输出仍有 \(L\) 个 token，每个 token 仍是 \(D\) 维。

### Self-attention 是否包含位置信息

仅看当前实现，答案是否定的。若交换输入 token 的顺序，attention 会随之等变地交换输出，但无法仅凭内容区分绝对或相对位置。Transformer 通常需要额外加入 positional encoding、relative position bias 或 RoPE。

### padding mask 应该屏蔽 Query 还是 Key

在 score matrix 中屏蔽 Key，可以阻止所有有效 Query 读取 padding token。padding Query 对应的输出通常还会在后续模块或 loss 计算中被忽略。若业务要求 padding Query 输出严格为零，需要额外处理 Query 维。

### dropout 后 attention weight 之和还是 1 吗

训练阶段不一定。softmax 后每行和为 1，但 dropout 会随机置零并缩放保留元素。推理阶段 dropout 关闭，attention weight 每行和仍为 1。

## 10. 运行示例

在当前目录执行：

```bash
python main.py
```

程序会打印输入形状、attention weight 及最终输出。由于线性层参数、输入张量和 dropout 都包含随机性，多次运行的具体数值可能不同，但维度保持不变：

```text
Input X shape: torch.Size([3, 4, 2])
attention_weights shape: torch.Size([3, 4, 4])
Output shape: torch.Size([3, 4, 2])
```

若要比较多次前向传播的数值结果，应设置随机种子，并在推理观察时调用：

```python
net.eval()
```

## 11. 结论

理解 self-attention 时，最有效的检查方式是持续追踪最后两个维度：

```text
[L, D] @ [D, L] -> [L, L]
[L, L] @ [L, D] -> [L, D]
```

第一次矩阵乘法构造 token 之间的关系，第二次矩阵乘法按该关系聚合 Value。面试中只要能准确解释这两次乘法、softmax 的维度和 mask 的广播方式，self-attention 的主体计算就已经清楚。
