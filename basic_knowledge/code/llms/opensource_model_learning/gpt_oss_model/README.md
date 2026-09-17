# GPT-OSS 最小教学复现

这个目录以 PyTorch 自包含地复现 Hugging Face GPT-OSS 的核心计算路径，用于理解模型结构，而不是加载或运行官方 GPT-OSS 权重。

## 保留的结构

- `configuration_gpt_oss.py`：`GptOssConfig` 保留 Hugging Face 原版公开字段、官方默认值和派生逻辑。默认配置对应 GPT-OSS-20B 级别，不能在普通设备直接实例化。
- `modeling_gpt_oss.py`：RMSNorm、GQA、attention sink、Top-k MoE、GPT-OSS 门控专家、decoder 和 causal LM head。
- `../utils/rope.py`：GPT-OSS 所用 YaRN 静态 inverse frequency 与 attention scaling、RoPE 旋转函数。
- `../utils/masks.py`：全局因果注意力和交替层使用的滑动窗口因果注意力。
- `../utils/cache.py`：用于 prefill 与增量解码的逐层 K/V 缓存。
- `../utils/training.py`：语言模型交叉熵之外的 Switch-style MoE 负载均衡辅助损失。

## 有意省略的内容

为保持代码集中于模型结构，此实现不包含 Transformers 的 `PreTrainedModel`、checkpoint 加载、自动注意力后端、Flash Attention、量化、分布式专家并行、通用文本生成器或 YaRN 的动态注册机制。

YaRN 本身完整保留 GPT-OSS 需要的静态参数路径：`factor`、`beta_fast`、`beta_slow`、`truncate`、`original_max_position_embeddings`，并在参数字典没有显式 `rope_theta` 时使用官方的 `150000.0` 默认值。

## 运行小模型示例

默认配置非常大。请显式创建小配置，或直接运行内置示例：

```bash
cd basic_knowledge/code/llms/opensource_model_learning/gpt_oss_model
python demo.py
```

示例会执行完整前向、KV cache 增量解码，以及一次 CE + MoE 辅助损失的反向传播。

## 运行测试

```bash
cd basic_knowledge/code/llms/opensource_model_learning/gpt_oss_model
python -m unittest discover -s tests -v
```

## 与原版的对应关系

| Hugging Face GPT-OSS | 本地教学实现 |
| --- | --- |
| `configuration_gpt_oss.py` | 同名轻量 `dataclass`，无 Transformers 基类 |
| `GptOssRotaryEmbedding` / rope utilities | `../utils/rope.py` 的 `YaRNRotaryEmbedding` |
| `GptOssAttention` | 同名模块，保留 GQA 与 sink |
| `GptOssTopKRouter`、`GptOssExperts`、`GptOssMLP` | 同名模块，保留稀疏 top-k 路由 |
| `DynamicCache` / masking utilities | `../utils/cache.py` 与 `../utils/masks.py` |
| `MixtralForCausalLM`、Transformers generation | 独立的 `GptOssForCausalLM` |
