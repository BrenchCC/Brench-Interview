# Grouped-Query Attention 原理与代码维度分析

Grouped-Query Attention 的核心改动很小:保留多组 Query head,但让多组 Query 共享更少组 Key/Value head。标准 Multi-Head Attention 中,Q、K、V 的 head 数通常相同;GQA 中,K/V head 数大于 1 且小于 Query head 数。若 K/V head 数压到 1,就是经典 Multi-Query Attention。

当前代码里的 `GroupedQueryAttention(hidden_dim = 128, nums_head = 8, nums_key_value_head = 4)` 是 GQA:8 个 Query head 被分成 4 组,每组共享同一个 K/V head。若把 `nums_key_value_head` 改成 1,才是经典 MQA。

## 问题定义

设输入为:

$$
X \in \mathbb{R}^{B \times S \times H}
$$

其中 `B` 是 batch size,`S` 是 sequence length,`H` 是 hidden dimension。代码中:

```python
batch_size, seq_length, hidden_dim = x.size()
```

在示例里:

```python
X = torch.randn(3, 2, 128)
net = GroupedQueryAttention(128, 8, 4)
```

所以:

| 符号 | 代码变量 | 示例值 | 含义 |
| --- | --- | ---: | --- |
| `B` | `batch_size` | 3 | batch size |
| `S` | `seq_length` | 2 | token 数 |
| `H` | `hidden_dim` | 128 | 每个 token 的 hidden size |
| `N_q` | `nums_head` | 8 | Query head 数 |
| `N_{kv}` | `nums_key_value_head` | 4 | Key/Value head 数 |
| `D` | `head_dim` | 16 | 单个 head 的维度,`H / N_q` |

代码里有两个约束:

```python
assert hidden_dim % nums_head == 0
assert nums_head % nums_key_value_head == 0
```

第一个约束保证每个 Query head 能拿到相同的 `head_dim`。第二个约束保证 K/V head 可以被均匀复制到 Query head 上。例如 `8 / 4 = 2`,表示每个 K/V head 服务 2 个 Query head。

## GQA 与标准 MHA 的差异

标准 MHA 会为每个 Query head 配一组 K/V head:

$$
Q, K, V \in \mathbb{R}^{B \times N_q \times S \times D}
$$

GQA 改成:

$$
Q \in \mathbb{R}^{B \times N_q \times S \times D}
$$

$$
K, V \in \mathbb{R}^{B \times N_{kv} \times S \times D}, \quad N_{kv} \le N_q
$$

注意力分数仍然要按 Query head 计算,最终 shape 仍是:

$$
\text{AttentionWeights} \in \mathbb{R}^{B \times N_q \times S \times S}
$$

区别在于 K/V 的来源少了。推理阶段做 KV Cache 时,缓存 K/V 的 head 数从 `N_q` 降为 `N_{kv}`。这就是 GQA 对长序列推理更有价值的地方:少存 K/V,少读 K/V,显存带宽压力更小。

当前实现为了写法直观,在计算 attention 前用 `repeat_interleave` 把 K/V 复制回 `N_q` 个 head。这个版本适合理解原理,但生产实现通常不会真的物理复制 K/V,而是在 kernel 内部处理 head 映射关系,否则会抵消一部分显存收益。

## 线性投影

代码先把输入 `x` 投影成 Q、K、V:

```python
query = self.query_proj(x)
key = self.key_proj(x)
value = self.value_proj(x)
```

对应模块定义:

```python
self.query_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)
self.key_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
self.value_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
```

`nn.Linear(in_features, out_features)` 对输入最后一维做线性变换。若输入是 `(B, S, H)`,它不会改变前两维,只把最后一维从 `H` 变成 `out_features`。

因此示例中的 shape 变化为:

| 张量 | 线性层输出维度 | 示例 shape |
| --- | ---: | --- |
| `query` | `nums_head * head_dim = 8 * 16 = 128` | `(3, 2, 128)` |
| `key` | `nums_key_value_head * head_dim = 4 * 16 = 64` | `(3, 2, 64)` |
| `value` | `nums_key_value_head * head_dim = 4 * 16 = 64` | `(3, 2, 64)` |

这里能看出参数量已经发生变化:Q 的投影仍输出 `hidden_dim`,K/V 的输出维度变小。标准 MHA 中,K/V 也会输出 128;当前代码中,K/V 只输出 64。

## 拆分 head 与维度转置

投影后,代码把最后一维拆成 `head` 和 `head_dim`:

```python
query = query.view(batch_size, seq_length, self.nums_head, self.head_dim).transpose(1, 2)
key = key.view(batch_size, seq_length, self.nums_key_value_head, self.head_dim).transpose(1, 2)
value = value.view(batch_size, seq_length, self.nums_key_value_head, self.head_dim).transpose(1, 2)
```

`view(...)` 只改变张量的逻辑形状,元素总数必须保持不变。以 `query` 为例,`(3, 2, 128)` 被解释成 `(3, 2, 8, 16)`。这一步没有做矩阵乘法,只是把原来的 hidden dimension 拆成 `nums_head * head_dim`。

`transpose(1, 2)` 交换第 1 维和第 2 维。代码希望 attention 计算时 head 维在 sequence 维之前,因此把 `(B, S, N, D)` 改成 `(B, N, S, D)`。

示例中的完整变化:

| 张量 | `view` 后 | `transpose(1, 2)` 后 |
| --- | --- | --- |
| `query` | `(3, 2, 8, 16)` | `(3, 8, 2, 16)` |
| `key` | `(3, 2, 4, 16)` | `(3, 4, 2, 16)` |
| `value` | `(3, 2, 4, 16)` | `(3, 4, 2, 16)` |

面试里可以直接说:attention 的常见计算布局是 `(batch, head, seq, head_dim)`,这样每个 head 都可以独立计算 `QK^T`。

## K/V head 复制

此时 `query` 有 8 个 head,`key/value` 只有 4 个 head。为了用普通矩阵乘法计算 attention,代码先把 K/V 复制到 8 个 head:

```python
key = key.repeat_interleave(self.nums_head // self.nums_key_value_head, dim = 1)
value = value.repeat_interleave(self.nums_head // self.nums_key_value_head, dim = 1)
```

`repeat_interleave(repeats, dim)` 会沿指定维度逐元素重复。这里 `dim = 1` 是 head 维,`repeats = 8 // 4 = 2`。所以:

$$
K: (3, 4, 2, 16) \rightarrow (3, 8, 2, 16)
$$

$$
V: (3, 4, 2, 16) \rightarrow (3, 8, 2, 16)
$$

复制关系可以理解为:

| Query head | 使用的原始 K/V head |
| ---: | ---: |
| 0 | 0 |
| 1 | 0 |
| 2 | 1 |
| 3 | 1 |
| 4 | 2 |
| 5 | 2 |
| 6 | 3 |
| 7 | 3 |

如果 `nums_key_value_head = 1`,那么 8 个 Query head 都共享同一个 K/V head,这就是经典 MQA。如果 `nums_key_value_head = nums_head`,`repeat_interleave` 的重复次数为 1,退化回标准 MHA 的 head 对齐形式。

## Attention 分数计算

代码中的核心计算是:

```python
attention_weights = torch.matmul(query, key.transpose(2,3)) / math.sqrt(self.head_dim)
```

`key.transpose(2, 3)` 把 K 从 `(B, N_q, S, D)` 变成 `(B, N_q, D, S)`。示例中:

$$
K: (3, 8, 2, 16) \rightarrow (3, 8, 16, 2)
$$

`torch.matmul` 对高维张量做批量矩阵乘法。它把最后两维当成矩阵维度,前面的维度按 batch 规则广播或逐块计算。因此:

$$
QK^T: (3, 8, 2, 16) \times (3, 8, 16, 2) \rightarrow (3, 8, 2, 2)
$$

最后一维的含义是:每个 query token 对所有 key token 的打分。除以 `sqrt(head_dim)` 是 scaled dot-product attention 的标准缩放,用于避免 `head_dim` 较大时点积值过大,导致 softmax 过早饱和。

## Mask 与 Softmax

代码保留了 mask 分支:

```python
if attention_mask is not None:
    attention_weights = attention_weights.masked_fill(0, float("-1e20"))

attention_weights = torch.softmax(attention_weights, dim = -1)
```

这里需要注意一个实现问题:`masked_fill` 的第一个参数应当是布尔 mask 张量,当前写成 `0` 并不能表达有效的 attention mask。更常见的写法是:

```python
attention_weights = attention_weights.masked_fill(attention_mask == 0, float("-1e20"))
```

实际使用时还要保证 `attention_mask` 能 broadcast 到 `(B, N_q, S, S)`。例如 causal mask 可以是 `(1, 1, S, S)`,padding mask 常见形状是 `(B, 1, 1, S)`。

`torch.softmax(attention_weights, dim = -1)` 沿最后一维归一化,也就是对每个 query token 的所有 key token 分数做 softmax。输出 shape 不变:

$$
(3, 8, 2, 2) \rightarrow (3, 8, 2, 2)
$$

归一化后,最后一维上的元素和为 1。

## 加权求和与输出投影

注意力权重乘以 value:

```python
output = attention_weights @ value
```

`@` 是 `torch.matmul` 的语法糖。此处:

$$
(3, 8, 2, 2) \times (3, 8, 2, 16) \rightarrow (3, 8, 2, 16)
$$

含义是每个 Query head 内,对 `S` 个 value token 做加权求和,得到每个 query token 的 head 输出。

接着代码把 head 维合并回 hidden dimension:

```python
output = output.transpose(1, 2).contiguous()
output = output.view(batch_size, seq_length, -1)
output = self.out_proj(output)
```

`output.transpose(1, 2)` 把 `(B, N_q, S, D)` 转回 `(B, S, N_q, D)`:

$$
(3, 8, 2, 16) \rightarrow (3, 2, 8, 16)
$$

`transpose` 后的张量通常不是连续内存布局。`contiguous()` 会返回一个内存连续的张量,保证后面的 `view` 可以按预期重新解释形状。

`view(batch_size, seq_length, -1)` 把 `N_q * D` 合并回 hidden dimension:

$$
(3, 2, 8, 16) \rightarrow (3, 2, 128)
$$

最后 `out_proj = nn.Linear(hidden_dim, hidden_dim)` 做一次输出投影,shape 不变:

$$
(3, 2, 128) \rightarrow (3, 2, 128)
$$

## 全流程维度表

以 `X = torch.randn(3, 2, 128)`、`nums_head = 8`、`nums_key_value_head = 4` 为例:

| 步骤 | Query shape | Key shape | Value shape | 说明 |
| --- | --- | --- | --- | --- |
| 输入 | `(3, 2, 128)` | `(3, 2, 128)` | `(3, 2, 128)` | 同一个 `x` 输入三组投影 |
| 线性投影 | `(3, 2, 128)` | `(3, 2, 64)` | `(3, 2, 64)` | K/V 输出维度减少 |
| 拆分 head | `(3, 2, 8, 16)` | `(3, 2, 4, 16)` | `(3, 2, 4, 16)` | `view` 拆最后一维 |
| 调整布局 | `(3, 8, 2, 16)` | `(3, 4, 2, 16)` | `(3, 4, 2, 16)` | `transpose(1, 2)` |
| 复制 K/V | `(3, 8, 2, 16)` | `(3, 8, 2, 16)` | `(3, 8, 2, 16)` | `repeat_interleave(..., dim = 1)` |
| 分数计算 | `(3, 8, 2, 16)` | `(3, 8, 16, 2)` | `(3, 8, 2, 16)` | `key.transpose(2, 3)` |
| attention weights | `(3, 8, 2, 2)` | - | - | `QK^T / sqrt(D)` |
| 加权求和 | - | - | - | `(3, 8, 2, 2) @ (3, 8, 2, 16) = (3, 8, 2, 16)` |
| 合并 head | - | - | - | `(3, 8, 2, 16) -> (3, 2, 8, 16) -> (3, 2, 128)` |
| 输出投影 | - | - | - | `(3, 2, 128)` |

## 复杂度与 KV Cache

从参数量看,标准 MHA 的 Q/K/V 投影都是 `H -> H`,三者参数量约为:

$$
3H^2
$$

当前实现中,Q 仍是 `H -> H`,K/V 是 `H -> N_{kv}D`。由于 `D = H / N_q`,K/V 的输出维度为:

$$
N_{kv}D = H \cdot \frac{N_{kv}}{N_q}
$$

因此 Q/K/V 投影参数量约为:

$$
H^2 + 2H^2 \cdot \frac{N_{kv}}{N_q}
$$

示例中 `N_{kv} / N_q = 4 / 8 = 1/2`,所以 K/V 投影参数量减半。若是经典 MQA,`N_{kv} = 1`,K/V 参数量会进一步下降。

更重要的是 KV Cache。自回归推理时,每生成一个 token 都要缓存历史 K/V。标准 MHA 的 K/V cache 规模与 `N_q` 成正比,GQA/MQA 与 `N_{kv}` 成正比:

$$
\text{KVCache} \propto B \times S \times N_{kv} \times D
$$

这也是面试中更值得强调的收益:GQA/MQA 主要优化推理阶段的 KV Cache 存储和读取,不是让 attention 分数矩阵从 `(B, N_q, S, S)` 变小。attention weights 仍然按 Query head 计算。

## PyTorch 方法速记

| 方法 | 在代码中的作用 | 维度影响 |
| --- | --- | --- |
| `x.size()` | 读取输入张量形状 | 不改变张量 |
| `nn.Linear` | 对最后一维做线性投影 | `(B, S, in) -> (B, S, out)` |
| `view` | 重新解释张量形状 | 元素总数不变 |
| `transpose(dim0, dim1)` | 交换两个维度 | 例如 `(B, S, N, D) -> (B, N, S, D)` |
| `repeat_interleave` | 沿某一维逐元素重复 | 当前用于把 K/V head 从 4 扩到 8 |
| `torch.matmul` | 批量矩阵乘法 | 使用最后两维做矩阵乘法 |
| `@` | `torch.matmul` 的语法糖 | 当前用于 attention weights 乘 value |
| `masked_fill` | 按布尔 mask 替换元素 | shape 不变 |
| `torch.softmax(..., dim = -1)` | 沿最后一维归一化 | shape 不变 |
| `contiguous` | 生成连续内存布局 | shape 不变 |

## 面试表达

可以这样回答:

> GQA 保留多个 Query head,但减少 Key/Value head 的数量。标准 MHA 中每个 Query head 都有独立的 K/V head;GQA 中一组 Query head 共享同一组 K/V。这样 attention weights 的 head 数仍然等于 Query head 数,但 K/V 投影参数量和推理时的 KV Cache 会下降。当前代码里 `nums_head = 8`,`nums_key_value_head = 4`,所以每 2 个 Query head 共享 1 个 K/V head。实现上先把 Q reshape 成 `(B, 8, S, D)`,K/V reshape 成 `(B, 4, S, D)`,再通过 `repeat_interleave` 沿 head 维复制成 `(B, 8, S, D)`,最后按标准 scaled dot-product attention 计算。

如果被追问为什么能加速推理,不要只说"计算量变少"。更准确的说法是:GQA/MQA 显著减少 KV Cache 的存储和读取压力,尤其在长上下文自回归解码时收益明显。attention 矩阵本身仍是 `(B, nums_head, seq_length, seq_length)`,并没有因为 K/V head 少了而直接变成 `(B, nums_key_value_head, seq_length, seq_length)`。
