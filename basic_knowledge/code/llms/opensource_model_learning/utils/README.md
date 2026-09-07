# Open-Source LLM Learning Shared Utils

该目录存放 GPT-OSS、Qwen3、Qwen3-MoE、Qwen3-VL、GLM4-MoE、GLM5-Next、Qwen3.5 与 Qwen3.5-MoE 的必要共享依赖，避免各模型复制 Transformers 的基础模块。

- `cache.py`：逐层 K/V cache。
- `configuration.py`：轻量配置基类、字段别名和层类型兼容转换。
- `masks.py`：全局因果与滑动窗口加性 mask。
- `rope.py`：GPT-OSS 的 YaRN 与 Qwen 系列标准 RoPE。
- `gated_delta.py`：Qwen3.5 的 Gated Delta Net、短卷积和递归 cache。
- `hyper_connection.py`：GLM5-Next 的 Manifold-Constrained Hyper-Connections 与 KDA forget gate。
- `vision.py`：Qwen3-VL 与 GLM5-Next 的 packed vision grid、位置与插值计算。
- `processing.py`：多模态 processor 使用的轻量组件接口和 token-count 数据类。
- `outputs.py`：因果语言模型与 MoE 的轻量输出数据类。
- `training.py`：MoE 的 Switch 风格负载均衡辅助损失。

这些模块仅替代 Transformers 的基础运行时；模型特有的配置、层和关键官方算法注释保留在各自模型目录中。
