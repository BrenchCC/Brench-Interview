# Qwen3 最小教学复现

这个目录以自包含 PyTorch 代码复现 Hugging Face Qwen3 的核心计算路径，目标是学习结构，不用于加载官方 checkpoint。

## 保留的结构

- [configuration_qwen3.py](configuration_qwen3.py)：保留 Qwen3-8B 风格的官方默认字段、KV head 回退和可选后段滑动窗口层逻辑。
- [modeling_qwen3.py](modeling_qwen3.py)：RMSNorm、GQA、Q/K per-head RMSNorm、标准 RoPE、SiLU-gated MLP、decoder、KV cache 与 Causal LM loss。
- [../utils](../utils)：与 GPT-OSS 共用的 K/V cache、因果/滑动掩码、RoPE、输出类型等最小依赖。

## 有意省略的内容

不包含 Transformers 的 `PreTrainedModel`、checkpoint 加载、自动注意力后端、Flash Attention、量化、张量并行、通用生成框架及任务分类头。

## 运行 toy 配置

默认配置为 Qwen3-8B 级别，不能在普通设备直接实例化。内置 demo 显式传入小配置：

```bash
cd basic_knowledge/code/llms/opensource_model_learning/qwen3_model
python demo.py
```

## 运行测试

```bash
cd basic_knowledge/code/llms/opensource_model_learning/qwen3_model
python -m unittest discover -s tests -v
```

## 与原版的对应关系

| Hugging Face Qwen3 | 本地教学实现 |
| --- | --- |
| `Qwen3Config` | 同名轻量 `dataclass`，无 Transformers 基类 |
| `Qwen3RMSNorm`、`Qwen3RotaryEmbedding` | 同名 RMSNorm 与共享 `RotaryEmbedding` |
| `Qwen3Attention` | 同名模块，保留 GQA、Q/K norm 和滑动窗口层 |
| `Qwen3MLP` | 同名 SiLU gate/up/down MLP |
| `DynamicCache` / masking utilities | 共享 `../utils/cache.py` 与 `../utils/masks.py` |
| `Qwen2ForCausalLM`、Transformers generation | 独立的 `Qwen3ForCausalLM` |
