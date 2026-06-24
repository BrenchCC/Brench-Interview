# Multi-Query Attention 原理与代码维度分析

Multi-Query Attention 的核心思想是:保留多个 Query head,但只保留 1 组 Key/Value head。标准 Multi-Head Attention 中,Q、K、V 的 head 数通常相同;MQA 中,所有 Query head 共享同一组 K/V。

当前代码里的 `MultiQueryAttention(hidden_dim = 128, nums_head = 8)` 固定:

```python
nums_key_value_head = 1
```

这和 `group_query_attention` 里的 GQA 不同。GQA 是 `1 < nums_key_value_head < nums_head`,例如 8 个 Query head 共享 4 个 K/V head;MQA 则是 8 个 Query head 共享 1 个 K/V head。

## 问题定义

设输入为:

$$
X \in \mathbb{R}^{B \times S \times H}
$$

其中 `B` 是 batch size,`S` 是 sequence length,`H` 是 hidden dimension。示例中:

```python
X = torch.randn(3, 2, 128)
net = MultiQueryAttention(hidden_dim = 128, nums_head = 8)
```

所以:

| 符号 | 代码变量 | 示例值 | 含义 |
| --- | --- | ---: | --- |
| `B` | `batch_size` | 3 | batch size |
| `S` | `seq_length` | 2 | token 数 |
| `H` | `hidden_dim` | 128 | 每个 token 的 hidden size |
| `N_q` | `nums_head` | 8 | Query head 数 |
| `N_{kv}` | `nums_key_value_head` | 1 | Key/Value head 数 |
| `D` | `head_dim` | 16 | 单个 head 的维度,`H / N_q` |

代码里要求:

```python
assert hidden_dim % nums_head == 0
```

这个约束保证每个 Query head 能拿到相同的 `head_dim`。示例中 `128 / 8 = 16`。

## MQA 与标准 MHA 的差异

标准 MHA 会为每个 Query head 配一组 K/V head:

$$
Q, K, V \in \mathbb{R}^{B \times N_q \times S \times D}
$$

MQA 改成:

$$
Q \in \mathbb{R}^{B \times N_q \times S \times D}
$$

$$
K, V \in \mathbb{R}^{B \times 1 \times S \times D}
$$

注意力分数仍然要按 Query head 计算,最终 shape 仍是:

$$
\text{AttentionWeights} \in \mathbb{R}^{B \times N_q \times S \times S}
$$

所以 MQA 不是把 attention weights 的 head 数变成 1,而是让所有 Query head 在计算时共享同一份 K/V。它主要节省的是 K/V 投影参数和推理阶段的 KV Cache。

## 线性投影

代码先把输入投影成 Q、K、V:

```python
query = self.query_proj(x)
key = self.key_proj(x)
value = self.value_proj(x)
```

对应模块定义:

```python
self.query_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)
self.key_proj = nn.Linear(hidden_dim, self.nums_key_value_head * self.head_dim)
self.value_proj = nn.Linear(hidden_dim, self.nums_key_value_head * self.head_dim)
```

由于 `nums_key_value_head = 1`,示例中的 shape 变化为:

| 张量 | 线性层输出维度 | 示例 shape |
| --- | ---: | --- |
| `query` | `8 * 16 = 128` | `(3, 2, 128)` |
| `key` | `1 * 16 = 16` | `(3, 2, 16)` |
| `value` | `1 * 16 = 16` | `(3, 2, 16)` |

这就是 MQA 和标准 MHA 参数量差异最直观的地方:Q 仍然输出 `hidden_dim`,但 K/V 只输出一个 head 的维度。

## 拆分 Head 与维度转置

投影后,代码把最后一维拆成 `head` 和 `head_dim`:

```python
query = query.view(batch_size, seq_length, self.nums_head, self.head_dim).transpose(1, 2)
key = key.view(batch_size, seq_length, self.nums_key_value_head, self.head_dim).transpose(1, 2)
value = value.view(batch_size, seq_length, self.nums_key_value_head, self.head_dim).transpose(1, 2)
```

示例中的完整变化:

| 张量 | `view` 后 | `transpose(1, 2)` 后 |
| --- | --- | --- |
| `query` | `(3, 2, 8, 16)` | `(3, 8, 2, 16)` |
| `key` | `(3, 2, 1, 16)` | `(3, 1, 2, 16)` |
| `value` | `(3, 2, 1, 16)` | `(3, 1, 2, 16)` |

attention 的常见计算布局是 `(batch, head, seq, head_dim)`,这样每个 head 可以独立计算 `QK^T`。

## K/V Head 复制

此时 `query` 有 8 个 head,`key/value` 只有 1 个 head。为了用普通矩阵乘法计算 attention,代码先把 K/V 复制到 8 个 head:

```python
key = key.repeat_interleave(self.nums_head, dim = 1)
value = value.repeat_interleave(self.nums_head, dim = 1)
```

所以:

$$
K: (3, 1, 2, 16) \rightarrow (3, 8, 2, 16)
$$

$$
V: (3, 1, 2, 16) \rightarrow (3, 8, 2, 16)
$$

复制关系可以理解为:

| Query head | 使用的原始 K/V head |
| ---: | ---: |
| 0 | 0 |
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 0 |
| 5 | 0 |
| 6 | 0 |
| 7 | 0 |

当前实现为了写法直观,物理复制了 K/V。生产实现通常不会真的复制 KV Cache,而是在 kernel 内部处理 head 映射关系,否则会抵消一部分显存收益。

## Attention 计算

代码中的核心计算是:

```python
attention_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
```

复制后的 `query` 和 `key` shape 分别是:

$$
Q: (3, 8, 2, 16)
$$

$$
K: (3, 8, 2, 16)
$$

`key.transpose(2, 3)` 后变成 `(3, 8, 16, 2)`,所以:

$$
QK^T: (3, 8, 2, 16) \times (3, 8, 16, 2) \rightarrow (3, 8, 2, 2)
$$

除以 `sqrt(head_dim)` 是 scaled dot-product attention 的标准缩放,用于避免点积值过大导致 softmax 饱和。

如果提供 `attention_mask`,代码会把 mask 为 0 的位置设成很小的值:

```python
attention_weights = attention_weights.masked_fill(attention_mask == 0, float("-1e20"))
```

实际使用时,`attention_mask` 需要能 broadcast 到 `(B, N_q, S, S)`。例如 causal mask 可以是 `(1, 1, S, S)`,padding mask 常见形状是 `(B, 1, 1, S)`。

接着沿最后一维做 softmax:

```python
attention_weights = torch.softmax(attention_weights, dim = -1)
```

输出 shape 不变,仍是 `(3, 8, 2, 2)`。

## 加权求和与输出投影

注意力权重乘以 value:

```python
output = attention_weights @ value
```

此时:

$$
(3, 8, 2, 2) \times (3, 8, 2, 16) \rightarrow (3, 8, 2, 16)
$$

然后把 head 维合并回 hidden dimension:

```python
output = output.transpose(1, 2).contiguous()
output = output.view(batch_size, seq_length, -1)
output = self.out_proj(output)
```

shape 变化为:

$$
(3, 8, 2, 16) \rightarrow (3, 2, 8, 16) \rightarrow (3, 2, 128)
$$

最后 `out_proj` 做一次输出投影,shape 保持 `(3, 2, 128)`。

## 全流程维度表

以 `X = torch.randn(3, 2, 128)`、`nums_head = 8`、`nums_key_value_head = 1` 为例:

| 步骤 | Query shape | Key shape | Value shape | 说明 |
| --- | --- | --- | --- | --- |
| 输入 | `(3, 2, 128)` | `(3, 2, 128)` | `(3, 2, 128)` | 同一个 `x` 输入三组投影 |
| 线性投影 | `(3, 2, 128)` | `(3, 2, 16)` | `(3, 2, 16)` | K/V 只保留 1 个 head |
| 拆分 head | `(3, 2, 8, 16)` | `(3, 2, 1, 16)` | `(3, 2, 1, 16)` | `view` 拆最后一维 |
| 调整布局 | `(3, 8, 2, 16)` | `(3, 1, 2, 16)` | `(3, 1, 2, 16)` | `transpose(1, 2)` |
| 复制 K/V | `(3, 8, 2, 16)` | `(3, 8, 2, 16)` | `(3, 8, 2, 16)` | `repeat_interleave(self.nums_head, dim = 1)` |
| attention weights | `(3, 8, 2, 2)` | - | - | `QK^T / sqrt(D)` |
| 加权求和 | - | - | - | `(3, 8, 2, 2) @ (3, 8, 2, 16) = (3, 8, 2, 16)` |
| 合并 head | - | - | - | `(3, 8, 2, 16) -> (3, 2, 8, 16) -> (3, 2, 128)` |
| 输出投影 | - | - | - | `(3, 2, 128)` |

## 复杂度与 KV Cache

从参数量看,标准 MHA 的 Q/K/V 投影都是 `H -> H`,三者参数量约为:

$$
3H^2
$$

MQA 中,Q 仍是 `H -> H`,K/V 是 `H -> D`。由于 `D = H / N_q`,Q/K/V 投影参数量约为:

$$
H^2 + 2H \cdot D = H^2 + 2H^2 \cdot \frac{1}{N_q}
$$

示例中 `N_q = 8`,所以 K/V 投影参数量从标准 MHA 的 `2H^2` 降为 `2H^2 / 8`。

更重要的是 KV Cache。自回归推理时,每生成一个 token 都要缓存历史 K/V。标准 MHA 的 K/V cache 规模与 `N_q` 成正比,MQA 与 1 个 K/V head 成正比:

$$
\text{KVCache}_{MQA} \propto B \times S \times 1 \times D
$$

这也是面试中更值得强调的收益:MQA 主要优化推理阶段的 KV Cache 存储和读取,尤其在长上下文自回归解码时收益明显。

## 面试表达

可以这样回答:

> MQA 保留多个 Query head,但所有 Query head 共享同一组 Key/Value head。标准 MHA 中每个 Query head 都有独立的 K/V head;MQA 中 K/V head 数固定为 1。当前代码里 `nums_head = 8`,`nums_key_value_head = 1`,所以 Q reshape 成 `(B, 8, S, D)`,K/V reshape 成 `(B, 1, S, D)`,再把 K/V 沿 head 维复制成 `(B, 8, S, D)` 来做标准 scaled dot-product attention。

如果被追问为什么能加速推理,不要只说"计算量变少"。更准确的说法是:MQA 显著减少 KV Cache 的存储和读取压力。attention 矩阵本身仍是 `(B, nums_head, seq_length, seq_length)`,并不会因为 K/V head 是 1 就直接变成 `(B, 1, seq_length, seq_length)`。
