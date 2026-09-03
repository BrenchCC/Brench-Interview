# Open-Source LLM Learning Shared Utils

该目录存放 `gpt_oss_model` 与 `qwen3_model` 的最小共享依赖，避免各模型复制 Transformers 的基础模块。

- `cache.py`：逐层 K/V cache。
- `masks.py`：全局因果与滑动窗口加性 mask。
- `rope.py`：GPT-OSS 的 YaRN 与 Qwen3 的标准 RoPE。
- `outputs.py`：两套模型的轻量输出数据类。
- `training.py`：GPT-OSS MoE 的负载均衡辅助损失。
