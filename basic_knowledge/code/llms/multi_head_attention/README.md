# Multi-Head Attention 基础原理与代码分析

本文对应 `mha.py` 和 `mha_with_kv_cache.py` 中的 multi-head attention 实现。前半部分先分析普通 MHA:multi-head attention 相比单头 self-attention 多了什么,代码里的每一步张量维度怎样变化,以及相关 PyTorch 方法在这里具体做了什么。后半部分再单独说明 KV Cache 版本的差异。

当前代码样例为:

```python
X = torch.randn(3, 2, 128)
mha = MultiHeadAttention(hidden_dim = 128, nums_head = 8)
output = mha(X, attention_mask)
```

因此:

| 符号    | 代码变量       | 当前样例取值 | 含义                          |
| ------- | -------------- | ------------ | ----------------------------- |
| `B`   | `batch_size` | `3`        | batch 中样本数量              |
| `L`   | `seq_len`    | `2`        | 每个样本的 token 数           |
| `D`   | `hidden_dim` | `128`      | 每个 token 的 hidden size     |
| `H`   | `nums_head`  | `8`        | attention head 数量           |
| `d_h` | `head_dim`   | `16`       | 每个 head 的维度,即 `D / H` |

输出仍然是 `(B, L, D)`。multi-head attention 不改变 token 数量,也不改变最终 hidden size。它改变的是中间计算方式:把 `D` 维表示拆成 `H` 个子空间,每个子空间独立做一次 scaled dot-product attention,再把结果拼回去。

## 1. 从单头到多头

单头 self-attention 的核心公式是:

```math
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
```

```math
A = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{D}}\right)
```

```math
O = AV
```

multi-head attention 只是把这套计算拆到多个 head 上。若 `hidden_dim = D`、`nums_head = H`、`head_dim = d_h`，通常要求:

```math
D = H \times d_h
```

每个 head 使用自己那一段 `Q`、`K`、`V` 做 attention:

```math
\text{head}_i =
\mathrm{softmax}\left(
\frac{Q_iK_i^T}{\sqrt{d_h}}
\right)V_i
```

最后拼接所有 head:

```math
O = \mathrm{Concat}(\text{head}_1,\ldots,\text{head}_H)W_O
```

这里缩放因子用的是 `sqrt(head_dim)`，不是 `sqrt(hidden_dim)`。原因很直接:每个 head 内部做点积时只使用 `head_dim` 个特征，点积方差主要由 `head_dim` 决定。

从面试角度看，multi-head attention 可以回答成一句话:它不是多做了几套完整的 `hidden_dim` attention，而是把同一个 hidden size 切成多个 head，在不同子空间里并行学习不同的 token 关系。

## 2. 模块初始化

代码:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head) -> None:
        super().__init__()
        self.nums_head = nums_head

        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // nums_head

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attention_dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
```

`query_proj`、`key_proj`、`value_proj` 都是 `nn.Linear(hidden_dim, hidden_dim)`。输入是 `(B, L, D)` 时，`nn.Linear` 只作用在最后一维:

```python
(B, L, D) -> (B, L, D)
```

在当前样例中:

```python
(3, 2, 128) -> (3, 2, 128)
```

这里容易有一个误解:既然有 8 个 head，是否需要 8 组独立的 `Linear`？当前实现没有显式写 8 个 `Linear`，而是用一个大的 `Linear(128, 128)` 一次性生成所有 head 的投影结果。后面的 `view` 会把最后一维拆成 `(8, 16)`。从参数形状看，它等价于把多个 head 的投影矩阵拼在一起计算。

一个工程细节也值得记住:代码里使用 `hidden_dim // nums_head`，但没有显式检查整除关系。如果 `hidden_dim` 不能被 `nums_head` 整除，`head_dim` 会被向下取整，后面的 `view(batch_size, seq_len, nums_head, head_dim)` 会因为元素数量对不上而报错。更稳妥的实现通常会加:

```python
assert hidden_dim % nums_head == 0
```

## 3. 前向传播中的维度变化

### 3.1 输入与线性投影

代码:

```python
batch_size, seq_len, hidden_dim = X.size()

Q = self.query_proj(X)
K = self.key_proj(X)
V = self.value_proj(X)
```

维度:

| 变量  | 形状          | 当前样例        |
| ----- | ------------- | --------------- |
| `X` | `(B, L, D)` | `(3, 2, 128)` |
| `Q` | `(B, L, D)` | `(3, 2, 128)` |
| `K` | `(B, L, D)` | `(3, 2, 128)` |
| `V` | `(B, L, D)` | `(3, 2, 128)` |

这一步只是在每个 token 的 hidden dimension 上做线性变换。token 之间还没有发生交互。真正的信息交换发生在后面的 `q_state @ k_state.transpose(-2, -1)`。

### 3.2 拆分 head

代码:

```python
q_state = Q.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
k_state = K.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
v_state = V.view(batch_size, seq_len, self.nums_head, self.head_dim).permute(0, 2, 1, 3)
```

以 `Q` 为例，先用 `view` 拆最后一维:

```python
Q:                         (B, L, D)
Q.view(..., H, d_h):       (B, L, H, d_h)
```

当前样例:

```python
(3, 2, 128) -> (3, 2, 8, 16)
```

然后用 `permute(0, 2, 1, 3)` 调整维度顺序:

```python
(B, L, H, d_h) -> (B, H, L, d_h)
```

当前样例:

```python
(3, 2, 8, 16) -> (3, 8, 2, 16)
```

为什么要把 `H` 放到 `L` 前面？因为后面希望每个 head 独立做 batch matrix multiplication。调整成 `(B, H, L, d_h)` 后，`B` 和 `H` 都可以看成批维度，最后两个维度才是矩阵乘法真正参与的维度。

拆分完成后:

| 变量        | 形状               | 当前样例          |
| ----------- | ------------------ | ----------------- |
| `q_state` | `(B, H, L, d_h)` | `(3, 8, 2, 16)` |
| `k_state` | `(B, H, L, d_h)` | `(3, 8, 2, 16)` |
| `v_state` | `(B, H, L, d_h)` | `(3, 8, 2, 16)` |

### 3.3 计算 attention score

代码:

```python
attention_weights = (
    q_state @ k_state.transpose(-2, -1) / math.sqrt(self.head_dim)
)
```

先看 `k_state.transpose(-2, -1)`。`k_state` 的形状是 `(B, H, L, d_h)`，`transpose(-2, -1)` 交换最后两个维度:

```python
k_state:                    (B, H, L, d_h)
k_state.transpose(-2, -1):  (B, H, d_h, L)
```

当前样例:

```python
(3, 8, 2, 16) -> (3, 8, 16, 2)
```

然后做矩阵乘法:

```python
q_state @ k_state.transpose(-2, -1)
(B, H, L, d_h) @ (B, H, d_h, L) -> (B, H, L, L)
```

当前样例:

```python
(3, 8, 2, 16) @ (3, 8, 16, 2) -> (3, 8, 2, 2)
```

`attention_weights[b, h]` 是第 `b` 个样本、第 `h` 个 head 的 `L x L` 分数矩阵。第 `i` 行表示第 `i` 个 query token 对所有 key token 的匹配分数。

除以 `math.sqrt(self.head_dim)` 后，维度不变:

```python
(B, H, L, L) -> (B, H, L, L)
```

变量名上要留意:此时它还是 attention score，经过 softmax 后才是概率意义上的 attention weight。

### 3.4 mask 逻辑

当前代码:

```python
if attention_mask is not None:
    attention_weights = attention_weights.masked_fill(
        attention_mask == 0, float("-1e20")
    )
```

这段代码在 softmax 前把不可见位置填成很小的负数。当前实现使用 `attention_mask == 0` 作为屏蔽条件,因此这里默认 `1` 表示可见、`0` 表示屏蔽。被填成 `-1e20` 的位置经过 softmax 后权重会接近 0。

mask 的形状也要和 `(B, H, L, L)` 对齐，或者能被 PyTorch broadcasting 到该形状。常见 padding mask 原始形状是 `(B, L)`，可以整理成:

```python
attention_mask = attention_mask[:, None, None, :]
```

形状变化:

```python
(B, L) -> (B, 1, 1, L)
```

它可以 broadcast 到 `(B, H, L, L)`，含义是所有 head、所有 query 位置都不能关注 padding key。若是 causal mask，常见形状是 `(1, 1, L, L)` 或 `(B, 1, L, L)`，用于屏蔽未来 token。

### 3.5 softmax 归一化

代码:

```python
attention_weights = torch.softmax(attention_weights, dim = -1)
```

`dim = -1` 表示沿最后一维做 softmax。对 `(B, H, L, L)` 来说，最后一维是 key token 维度，所以每个 query token 都会得到一行概率分布:

```python
attention_weights[b, h, i, :].sum() == 1
```

维度不变:

```python
(B, H, L, L) -> (B, H, L, L)
```

当前样例:

```python
(3, 8, 2, 2) -> (3, 8, 2, 2)
```

如果 mask 正确生效，被屏蔽位置在 softmax 后应接近 0。这里说「接近」是因为计算机里是浮点数；工程实现通常依赖足够小的负数让 `exp(score)` 几乎为 0。

### 3.6 attention dropout

代码:

```python
attention_weights = self.attention_dropout(attention_weights)
```

`nn.Dropout(0.1)` 在训练模式下会随机把部分 attention weight 置 0，并对保留值做缩放，使期望保持不变。它不改变张量形状:

```python
(B, H, L, L) -> (B, H, L, L)
```

当前代码在 `__main__` 里没有调用 `mha.eval()`，因此直接运行脚本时 dropout 会生效，每次输出可能不同。面试里如果被追问「为什么同一个输入多次输出不一样」，这里就是原因之一。

### 3.7 加权汇总 value

代码:

```python
output_state = attention_weights @ v_state
```

矩阵乘法维度:

```python
(B, H, L, L) @ (B, H, L, d_h) -> (B, H, L, d_h)
```

当前样例:

```python
(3, 8, 2, 2) @ (3, 8, 2, 16) -> (3, 8, 2, 16)
```

这一步的含义是:对每个样本、每个 head、每个 query token，用它对应的 attention probability 对所有 value 向量做加权求和。输出的第 `i` 个 token 已经混入了它在该 head 下关注到的上下文信息。

### 3.8 拼接多个 head

代码:

```python
output_state = output_state.permute(0, 2, 1, 3).contiguous()
output = output_state.view(batch_size, seq_len, -1)
```

先把 head 维度移回 token 维度之后:

```python
(B, H, L, d_h) -> (B, L, H, d_h)
```

当前样例:

```python
(3, 8, 2, 16) -> (3, 2, 8, 16)
```

然后用 `view(batch_size, seq_len, -1)` 把 `H` 和 `d_h` 合并:

```python
(B, L, H, d_h) -> (B, L, H * d_h) -> (B, L, D)
```

当前样例:

```python
(3, 2, 8, 16) -> (3, 2, 128)
```

这里的 `contiguous()` 不是数学操作，而是内存布局处理。`permute` 只改变 tensor 的 stride 视图，底层内存不一定连续。`view` 要求 tensor 在内存中连续，因此在 `permute` 后接 `contiguous()` 是常见写法。若改用 `reshape`，很多情况下可以不手动调用 `contiguous()`，因为 `reshape` 会在必要时创建连续副本。

### 3.9 输出投影

代码:

```python
output = self.out_proj(output)
```

`out_proj` 仍然是 `nn.Linear(hidden_dim, hidden_dim)`:

```python
(B, L, D) -> (B, L, D)
```

当前样例:

```python
(3, 2, 128) -> (3, 2, 128)
```

这一层的作用不是改变维度，而是混合不同 head 拼接后的信息。如果没有 `out_proj`，多个 head 只是简单拼在一起；加上输出投影后，模型可以学习如何重新组合这些子空间。

## 4. 完整维度流

| 步骤            | 表达式                                  | 形状变化                                            | 当前样例                                          |
| --------------- | --------------------------------------- | --------------------------------------------------- | ------------------------------------------------- |
| 输入            | `X`                                   | `(B, L, D)`                                       | `(3, 2, 128)`                                   |
| Q 投影          | `query_proj(X)`                       | `(B, L, D)`                                       | `(3, 2, 128)`                                   |
| K 投影          | `key_proj(X)`                         | `(B, L, D)`                                       | `(3, 2, 128)`                                   |
| V 投影          | `value_proj(X)`                       | `(B, L, D)`                                       | `(3, 2, 128)`                                   |
| 拆 Q head       | `Q.view(B, L, H, d_h)`                | `(B, L, D) -> (B, L, H, d_h)`                     | `(3, 2, 128) -> (3, 2, 8, 16)`                  |
| 调整 Q 维度     | `permute(0, 2, 1, 3)`                 | `(B, L, H, d_h) -> (B, H, L, d_h)`                | `(3, 2, 8, 16) -> (3, 8, 2, 16)`                |
| K 转置          | `k_state.transpose(-2, -1)`           | `(B, H, L, d_h) -> (B, H, d_h, L)`                | `(3, 8, 2, 16) -> (3, 8, 16, 2)`                |
| attention score | `q_state @ k_state.transpose(-2, -1)` | `(B, H, L, d_h) @ (B, H, d_h, L) -> (B, H, L, L)` | `(3, 8, 2, 16) @ (3, 8, 16, 2) -> (3, 8, 2, 2)` |
| 缩放            | `/ sqrt(d_h)`                         | `(B, H, L, L)`                                    | `(3, 8, 2, 2)`                                  |
| mask            | `masked_fill(...)`                    | `(B, H, L, L)`                                    | `(3, 8, 2, 2)`                                  |
| 归一化          | `softmax(dim = -1)`                   | `(B, H, L, L)`                                    | `(3, 8, 2, 2)`                                  |
| dropout         | `Dropout(0.1)`                        | `(B, H, L, L)`                                    | `(3, 8, 2, 2)`                                  |
| 加权汇总        | `attention_weights @ v_state`         | `(B, H, L, L) @ (B, H, L, d_h) -> (B, H, L, d_h)` | `(3, 8, 2, 2) @ (3, 8, 2, 16) -> (3, 8, 2, 16)` |
| 还原维度顺序    | `permute(0, 2, 1, 3)`                 | `(B, H, L, d_h) -> (B, L, H, d_h)`                | `(3, 8, 2, 16) -> (3, 2, 8, 16)`                |
| 拼接 head       | `view(B, L, -1)`                      | `(B, L, H, d_h) -> (B, L, D)`                     | `(3, 2, 8, 16) -> (3, 2, 128)`                  |
| 输出投影        | `out_proj(output)`                    | `(B, L, D) -> (B, L, D)`                          | `(3, 2, 128) -> (3, 2, 128)`                    |

## 5. PyTorch 方法说明

| 方法                                  | 当前代码中的作用                  | 面试中应说明的点                                                      |
| ------------------------------------- | --------------------------------- | --------------------------------------------------------------------- |
| `nn.Linear(hidden_dim, hidden_dim)` | 对输入最后一维做线性投影          | 输入可以是三维张量 `(B, L, D)`，只要最后一维等于 `in_features`    |
| `X.size()`                          | 读取输入形状                      | 返回 `(batch_size, seq_len, hidden_dim)`                            |
| `view(B, L, H, d_h)`                | 把 hidden dimension 拆成多个 head | 要求元素总数不变，因此 `D` 必须等于 `H * d_h`                     |
| `permute(0, 2, 1, 3)`               | 调整维度顺序                      | 从 `(B, L, H, d_h)` 变成 `(B, H, L, d_h)`，方便每个 head 独立计算 |
| `transpose(-2, -1)`                 | 交换最后两个维度                  | 把 K 从 `(B, H, L, d_h)` 变成 `(B, H, d_h, L)`                    |
| `@`                                 | 执行矩阵乘法                      | 对四维张量来说，前面的 `(B, H)` 是批维，最后两维参与矩阵乘法        |
| `math.sqrt(self.head_dim)`          | 缩放 attention score              | 缩放尺度来自每个 head 的维度，不是总 hidden size                      |
| `masked_fill(mask == 0, value)`     | 将 mask 命中的位置替换为极小值    | 通常在 softmax 前执行，使不可见位置权重接近 0                         |
| `torch.softmax(..., dim = -1)`      | 沿 key 维归一化                   | 每个 query token 对所有 key token 的概率和为 1                        |
| `nn.Dropout(0.1)`                   | 随机丢弃部分 attention weight     | 训练模式生效，`eval()` 模式关闭                                     |
| `contiguous()`                      | 让 tensor 内存连续                | 常接在 `permute` 后面，保证后续 `view` 可用                       |
| `view(B, L, -1)`                    | 合并 head 维和 head_dim 维        | `-1` 让 PyTorch 自动推断最后一维大小                                |
| `register_buffer(..., persistent = False)` | 注册非参数缓存             | tensor 会随模块迁移设备,但不会写入 `state_dict`                   |
| `torch.cat(..., dim = 2)`           | 拼接历史 K/V 与当前 K/V           | KV Cache 中第 2 维是序列长度维度                                     |

## 6. 当前实现的边界

这份代码适合面试复习和理解主干流程，但距离完整 transformer attention 仍有一些差距:

- 没有显式检查 `hidden_dim % nums_head == 0`，真实实现应当在初始化时检查。
- 当前实现只有 self-attention，因为 `Q`、`K`、`V` 都来自同一个 `X`。cross-attention 中，`Q` 通常来自 decoder hidden states，`K` 和 `V` 来自 encoder hidden states。
- `mha.py` 没有 residual connection、layer norm、FFN、position encoding 或 KV Cache。`mha_with_kv_cache.py` 补了 KV Cache,但仍只覆盖 attention 子层本身。
- `float("-1e20")` 在 fp32 中通常能工作；混合精度或不同 dtype 下，工程实现更常用框架内部的 mask 处理，或者使用和 dtype 匹配的最小值。

这些点不影响学习主流程，反而适合用来训练代码审阅能力。面试中主动说明 demo 实现的边界，比单纯背公式更有说服力。

## 7. KV Cache 版本解析

`mha_with_kv_cache.py` 的主干计算和普通 MHA 相同。差异集中在两个地方:初始化时注册 `key_cache`、`value_cache`,前向传播时允许把历史 K/V 拼到当前 K/V 前面。它解决的是自回归推理中的重复计算问题。

### 7.1 为什么只缓存 K 和 V

生成任务通常按 token 递增。假设已经处理过 `L_past` 个 token,下一步只输入 `L_cur` 个新 token。普通实现如果每一步都把完整上下文重新喂进去,会重复计算旧 token 的 `Q`、`K`、`V` 和旧 query 对应的 attention 输出。

KV Cache 的判断很简单:旧 token 的 hidden states 在当前层已经确定后,由它们投影得到的 K/V 不会因为新 token 到来而改变。新 token 的 query 需要看历史 K/V,但历史 token 的 query 输出不需要在单步解码时重新算。因此缓存 K/V 就够了,Q 只为当前输入重新计算。

代码接口是:

```python
def forward(self, x, attention_mask = None, past_kv_cache = None, store_kv_cache = False):
```

其中 `x` 是当前 step 的输入,不一定是完整序列。若是单 token 解码,它的形状通常是:

```python
x: (B, 1, D)
```

当前示例仍使用:

```python
X = torch.randn(3, 2, 128)
```

因此这里的 `seq_len = 2` 可以理解为一次传入 2 个当前 token。为了避免和历史长度混淆,下面记作 `L_cur`。

### 7.2 缓存的注册方式

初始化中新增了两行:

```python
self.register_buffer("key_cache", None, persistent = False)
self.register_buffer("value_cache", None, persistent = False)
```

`register_buffer` 注册的是非参数状态。它不会被 optimizer 更新,但会跟随模块执行 `.to(device)`、`.cuda()` 这类设备迁移。`persistent = False` 表示这两个缓存不写入 `state_dict`。这很合理:KV Cache 是一次推理过程中的临时状态,不是模型权重。

当前 forward 最终仍然显式返回 `(k_state, v_state)`:

```python
return output, (k_state, v_state)
```

所以更推荐调用方把返回的 cache 继续传回 `past_kv_cache`。模块内部的 `self.key_cache`、`self.value_cache` 只有在 `store_kv_cache = True` 时更新,当前代码没有在下一次 forward 中自动读取它们。

### 7.3 无历史 cache 的第一次调用

没有传入 `past_kv_cache` 时,KV Cache 版本退化成普通 MHA:

```python
output, (k_state, v_state) = mha(
    X,
    attention_mask = attention_mask,
    past_kv_cache = None,
    store_kv_cache = True
)
```

投影和拆 head 后:

| 变量        | 形状                       | 当前样例          |
| ----------- | -------------------------- | ----------------- |
| `q_state` | `(B, H, L_cur, d_h)`     | `(3, 8, 2, 16)` |
| `k_state` | `(B, H, L_cur, d_h)`     | `(3, 8, 2, 16)` |
| `v_state` | `(B, H, L_cur, d_h)`     | `(3, 8, 2, 16)` |

attention score 的形状仍是:

```python
(B, H, L_cur, d_h) @ (B, H, d_h, L_cur) -> (B, H, L_cur, L_cur)
```

当前样例:

```python
(3, 8, 2, 16) @ (3, 8, 16, 2) -> (3, 8, 2, 2)
```

返回的 `k_state`、`v_state` 就是后续 step 的历史 cache。若 `store_kv_cache = True`,它们也会被写入模块 buffer:

```python
self.key_cache = k_state
self.value_cache = v_state
```

### 7.4 传入历史 cache 后的拼接

当 `past_kv_cache` 不为空时,代码会取出历史 K/V:

```python
past_key, past_value = past_kv_cache
k_state = torch.cat([past_key, k_state], dim = 2)
v_state = torch.cat([past_value, v_state], dim = 2)
```

这里 `dim = 2` 是序列长度维。拆 head 后的 K/V 形状是 `(B, H, L, d_h)`,第 0 维是 batch,第 1 维是 head,第 2 维才是 token 位置。

若历史 cache 形状为:

```python
past_key:   (B, H, L_past, d_h)
past_value: (B, H, L_past, d_h)
```

当前 K/V 形状为:

```python
k_state: (B, H, L_cur, d_h)
v_state: (B, H, L_cur, d_h)
```

拼接后得到:

```python
k_state: (B, H, L_past + L_cur, d_h)
v_state: (B, H, L_past + L_cur, d_h)
```

当前 `__main__` 里的第二个测试构造了:

```python
k_state = torch.randn(3, 8, 2, 16)
v_state = torch.randn(3, 8, 2, 16)
past_kv_cache = (k_state, v_state)
```

同时当前输入 `X` 的 `L_cur = 2`。因此拼接后的 K/V 形状是:

```python
(3, 8, 2, 16) + (3, 8, 2, 16) -> (3, 8, 4, 16)
```

Q 不拼接历史,仍只来自当前输入:

```python
q_state: (3, 8, 2, 16)
```

attention score 形状随之变成:

```python
(B, H, L_cur, d_h) @ (B, H, d_h, L_past + L_cur)
-> (B, H, L_cur, L_past + L_cur)
```

当前样例:

```python
(3, 8, 2, 16) @ (3, 8, 16, 4) -> (3, 8, 2, 4)
```

这也是阅读 KV Cache 代码时最容易漏掉的变化:attention score 不再是 `(B, H, L, L)`,而是 `(B, H, L_cur, L_total)`,其中 `L_total = L_past + L_cur`。

### 7.5 输出形状为什么不随 cache 变长

虽然 K/V 的长度变成了 `L_past + L_cur`,输出长度仍然跟当前 query 长度一致。矩阵乘法是:

```python
attention_weights @ v_state
(B, H, L_cur, L_total) @ (B, H, L_total, d_h)
-> (B, H, L_cur, d_h)
```

也就是说,每个当前 query token 可以关注历史和当前的所有 key token,但本次 forward 只产出当前输入对应的 hidden states。后面的拼接 head 和输出投影仍是:

```python
(B, H, L_cur, d_h) -> (B, L_cur, H, d_h) -> (B, L_cur, D)
```

当前第二个测试中:

```python
output:  (3, 2, 128)
k_state: (3, 8, 4, 16)
v_state: (3, 8, 4, 16)
```

输出没有变成 `(3, 4, 128)`。KV Cache 扩展的是可被当前 query 访问的 key/value 序列,不是本次输出的 query 序列。

### 7.6 mask 与 causal 约束

KV Cache 版本的 mask 要对齐新的 score 形状:

```python
attention_weights: (B, H, L_cur, L_total)
```

padding mask 如果原始形状是 `(B, L_total)`,可以整理成:

```python
attention_mask = attention_mask[:, None, None, :]
```

它会 broadcast 到 `(B, H, L_cur, L_total)`。

causal mask 要更小心。单 token 解码时 `L_cur = 1`,当前 token 前面只有历史 token 和它自己,通常不需要额外屏蔽未来位置。若一次传入多个新 token,例如 `L_cur = 2`,第一个当前 token 不能看到第二个当前 token,此时 causal mask 应覆盖 `(L_cur, L_total)` 这个矩形区域,而不是简单复用普通训练里的 `(L_total, L_total)` 后不检查切片。

当前代码把 `attention_mask` 作为外部参数传入,没有在模块内部自动构造 causal mask。面试里可以明确说:KV Cache 只解决历史 K/V 复用,不自动保证自回归可见性；可见性仍由 mask 负责。

### 7.7 KV Cache 的收益和边界

KV Cache 的收益来自避免重复处理旧 token。以单 token 解码为例,第 `t` 步如果不用 cache,常见写法会重新计算长度为 `t` 的整段上下文,attention score 是 `(B, H, t, t)`；使用 cache 后,当前 step 只计算新 token 的 query 和 K/V,score 是 `(B, H, 1, t)`。旧 token 的 K/V 直接复用。

这不表示 attention 的序列长度成本消失了。新 token 仍要看所有历史 key,所以单步 attention 仍随当前总长度线性增长；只是不会反复为旧 query 重新计算输出。生成越长,cache 占用的显存也越多,因为每一层都要保存 K/V:

```python
num_layers * 2 * B * H * L_total * d_h
```

这里的 `2` 对应 key 和 value。实际部署还要乘以 dtype 字节数,例如 fp16/bf16 通常是 2 bytes。

## 8. 面试回答模板

如果被问「multi-head attention 的计算流程是什么」，可以按这个顺序回答:

1. 输入 `X` 的形状是 `(B, L, D)`，先通过三组线性层得到 `Q`、`K`、`V`，形状仍是 `(B, L, D)`。
2. 将最后一维拆成 `H` 个 head，每个 head 的维度是 `d_h = D / H`，再调整成 `(B, H, L, d_h)`。
3. 每个 head 内部计算 `QK^T / sqrt(d_h)`，得到 `(B, H, L, L)` 的 score。这里 `L x L` 表示每个 token 对所有 token 的匹配关系。
4. 如果有 mask，需要在 softmax 前把不可见位置填成很小的负数，然后沿最后一维做 softmax，得到每个 query 对所有 key 的概率分布。
5. 用 attention weight 乘以 `V`，得到 `(B, H, L, d_h)`，再把多个 head 拼接回 `(B, L, D)`。
6. 最后经过 `out_proj`，让不同 head 的信息重新混合，输出仍是 `(B, L, D)`。

一句判断可以放在结尾:multi-head attention 的价值在于让模型在多个低维子空间里并行建立 token 之间的动态路由；它的主要代价仍然来自 `L x L` attention score，时间和显存都随序列长度平方增长。

如果继续问「KV Cache 改了什么」,可以补充:

1. 普通 MHA 每次都根据当前输入计算 `Q`、`K`、`V`；KV Cache 版本会把历史 `K`、`V` 保存下来,下一步只计算当前 token 的 `Q`、`K`、`V`。
2. 有历史 cache 时,`K`、`V` 会沿序列长度维拼接:从 `(B, H, L_cur, d_h)` 变成 `(B, H, L_past + L_cur, d_h)`。
3. `Q` 不拼接历史,所以 attention score 从普通的 `(B, H, L, L)` 变成 `(B, H, L_cur, L_past + L_cur)`。
4. 输出长度仍等于当前输入长度,即 `(B, L_cur, D)`；返回的 cache 长度才会增长到 `L_past + L_cur`。
5. KV Cache 不替代 mask。padding mask 或 causal mask 仍要和 `(B, H, L_cur, L_total)` 对齐。
