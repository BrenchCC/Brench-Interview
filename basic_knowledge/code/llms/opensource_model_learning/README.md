# Open-source model learning ports

本目录以 Hugging Face Transformers 的当前模型源文件为参照，保留配置字段、核心层次、关键算法注释和 PyTorch 参数布局；同时去除 `PreTrainedModel`、自动 attention 后端、分布式 expert parallel、权重下载和通用 `generate` 运行时。

| 本地目录 | 对应官方模块 | 学习重点 |
| --- | --- | --- |
| `gpt_oss_model` | `models/gpt_oss` | YaRN、GQA、attention sink、交替滑动注意力、Top-k MoE |
| `qwen3_model` | `models/qwen3` | Q/K RMSNorm、GQA、RoPE、因果 decoder |
| `qwen3_moe_model` | `models/qwen3_moe` | packed expert 权重和 softmax Top-k routing |
| `qwen3_vl_model` | `models/qwen3_vl` | Conv3D patch、packed vision attention、视觉 token processor |
| `glm4_moe_model` | `models/glm4_moe` | partial RoPE、grouped sigmoid router、shared expert |
| `glm5_next_model` | `models/glm5_next` | KDA、DSA indexer、MLA、mHC、视觉塔与多模态 processor |
| `qwen3_5_model` | `models/qwen3_5` | hybrid Gated Delta Net / full attention decoder |
| `qwen3_5_moe_model` | `models/qwen3_5_moe` | Qwen3.5 hybrid decoder 上的 packed MoE |

所有公共依赖放在 `utils/`：配置生命周期、缓存、RoPE、mask、MoE loss、Gated Delta、mHC、视觉网格以及 processor 基类均只保留这些模型需要的最小实现。

官方默认配置对应大规模模型，不应在普通设备上直接实例化。请像各模型的测试一样显式传入 toy 配置，例如：

```python
from basic_knowledge.code.llms.opensource_model_learning.gpt_oss_model import GptOssConfig, GptOssForCausalLM

config = GptOssConfig(
    vocab_size = 128,
    hidden_size = 64,
    intermediate_size = 128,
    num_hidden_layers = 2,
    num_attention_heads = 4,
    num_key_value_heads = 2,
    num_local_experts = 4,
)
model = GptOssForCausalLM(config)
```

这些端口的目标是结构、张量形状和核心算法可读、可验证；它们不保证加载官方 checkpoint 或与官方高性能 kernel 逐 token 数值一致。
