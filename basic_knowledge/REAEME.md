# 面试题仓库

这是一个用于整理和管理面试题的仓库。

## 简介

本仓库收集和整理各类面试相关的资料和题目，帮助准备技术面试。

## 内容

- **大模型面试题**: 包含大模型相关的面试问答题目
- 持续更新中...

## 目录结构

```
interview/
├── data/          # 面试资料文件
└── README.md      # 项目说明
```

## 使用方法

1. 浏览 `data/` 目录查看面试相关资料
2. 根据需要学习和复习相关内容

## 说明

本仓库仅供个人学习和面试准备使用。

# 面试题分类

## Transformer 基础与核心组件

聚焦 Transformer 架构本质、注意力机制、归一化、FFN 等基础模块的原理与特性。
涵盖 Tokenization 训练与方法、位置编码原理、长度外推技术的核心考点。

- 模型结构与基本原理

  - 简述 Transformer 的基本结构和原理
  - Transformer 为什么使用多头注意力机制
  - Transformer 为何让 Q（查询）和 K（键）使用独立的权重矩阵，为什么需要 Q、K、V 三个矩阵
  - 为什么在 attention 中要进行 scaled（为什么除以 √d_k）
  - Transformer 的位置编码作用及局限性
  - Transformer 相比 RNN、LSTM 的优势何在
  - Transformer 在哪里做了权重共享
- 注意力机制深入

  - Multi-head Latent Attention（多头隐变量注意力）
  - Linear Attention（线性注意力）
  - Cross-Attention（交叉注意力）
  - Sparse Attention（稀疏注意力）
  - MLA（Multi-Linear Attention）
  - 不同 Attention 之间的区别
- 归一化与正则化机制

  - 为什么 Transformer 用 LayerNorm 而不用 BatchNorm
  - RMSNorm 正则化有什么好处和优势
  - Group Normalization（GN）和 Instance Normalization（IN）的区别
  - 不同的 Normalization 之间有什么区别
- 激活函数与非线性设计

  - Transformer 前馈神经网络用的是什么激活函数
  - SwiGLU 激活函数的原理
- 模型计算与参数分析

  - Transformer 模型中，最占用参数的是 MLP 层吗？
  - Transformer 模型中计算量（FLOPs）的分析
  - Transformer 模型中计算量（FLOPs）和参数的表格分析
- 训练与损失函数

  - 什么是交叉熵损失函数？大模型的哪里有交叉熵损失函数？
- 长度外推与扩展机制

  - 什么是长度外推
  - ALiBi（Attention with Linear Biases）的思路是什么
  - NTK 长度外推方法是什么

## 特定模型与结构变体

围绕经典模型（BERT/GLM/Llama 等）、模型架构类型（Encoder-only 等）、MOE 等特殊结构的考点。

- 模型结构与任务类型相关

  - CLM和MLM分别是什么,有什么区别?
  - GLM的自回归空白填充方法与BERT中的使用遮蔽语言模型(MLM)有什么不同,为自然语言理解(NLU)任务带来了哪些优势
  - Encoder-only、Decoder-only和Encoder-Decoder的模型分别有什么区别,怎么运用?
  - 为什么现在的大语言模型都用Decoder-only
- 模型架构与代表模型

  - 介绍一下Bert的结构
  - 介绍一下Llama的结构
  - 介绍一下 Qwen 通义千问的结构
  - Qwen2 有哪些提升？
  - 介绍一下 Llama3.1 的创新
- MOE 与模型结构变体

  - 什么是 MOE (混合专家模型)
  - Dense 和 MOE 模型的区别
- 最新模型与技术创新

  - DeepSeekV3 有啥技术特点？
  - DeepSeek-R1-Zero 里的 Zero 含义？
  - DeepSeek-R1-Zero 有什么弊端？

## 训练机制与优化技巧类

聚焦模型分布式训练技术，模型训练阶段的工程优化与稳定性问题。

- 混合精度训练相关
  - 解释一下混合精度的原理
  - 为什么需要混合精度训练？
  - 训练的时候用 float16、bfloat16 还是 float32, 为什么？
  - 怎么解决训练使用 float16 导致溢出的问题
- 训练优化与工具相关
  - DeepSpeed 中 ZeRO 系列的原理
  - 训练时数据长度不一致怎么办，以及如何优化训练速度
- 迁移学习与增量预训练相关
  - 了解迁移学习吗？大模型中是怎么运用迁移学习的？
  - 为什么需要增量预训练？
  - 增量预训练的过程当中，loss 上升正常吗？
  - 在增量预训练过程中如何设置学习率 (learning rate, LR)?
- 模型训练核心技术问题
  - Llama 模型在训练过程中如何处理梯度消失和梯度爆炸的问题
  - 什么是 warmup_ratio? 训练过程中怎么设置？

## sft和lora等参数高效微调技术

围绕 “低参量微调大模型” 的技术，涵盖各类轻量化微调方法的原理、初始化与差异。

- Lora及其变体相关
  - 解释一下Lora 的原理
  - Lora 是怎样进行初始化矩阵的
  - 解释一下AdaLora 和 QLora 的原理
- Adapter及相关参数高效微调技术
  - 解释一下 Adapter
  - 介绍几种常⻅的 Adapter
  - 解释一下 prefix-tuning
  - 解释一下 P-tuning
  - 解释一下 Prompt-tuning
- 预训练、微调及SFT相关
  - 预训练和微调任务有什么区别，两者的目的
  - SFT 后会出现哪些问题

## 强化学习

涉及训练过程中的技术优化、梯度问题、学习率设置，以及强化学习在大模型对齐中的应用。

- 强化学习基础概念

  - 解释下强化学习
  - 强化学习算法是怎么分类的
  - 强化学习中策略函数和值函数是什么
  - OpenAI 对齐为什么要用强化学习，别的方法不行吗
- 强化学习代表算法

  - PPO 的原理
  - DPO 的原理
  - 分析 DPO 和 PPO 的区别
- DPO 相关细节

  - DPO 的正负样本怎么构造
  - DPO 的 loss 是什么
  - 怎么理解 DPO 的损失函数
  - DPO 的微调流程
- 强化学习在实际大模型中的应用

  - DeepSeek-R1-Zero 中的强化学习原理
  - 在 DeepSeek-R1 上 PRM 和 MCTS 是否有用

## 推理优化与效率提升

聚焦大模型推理阶段的速度、显存优化技术，以及推理中的常见问题解决。

- 核心技术原理
  - 解释一下 KVcache 的原理
  - 介绍一下 FlashAttention 的原理
  - 为什么大模型用 GQA (Group Query Attention)
- 推理加速方案
  - 介绍一下 VLLM 的加速方法
  - 什么是 Prefill-Decode 分离？为什么分离？
  - 为什么要优化 KV cache
- 推理相关问题与瓶颈
  - 如何缓解大模型 inference 的时候的重复问题
  - 为什么推理速度受限于显存带宽

## 模型评估、工具与概念辨析

包含模型评估指标、工具使用、关键概念（端到端 / 冷启动等）的定义与辨析。

- 从 huggingface 下载模型时有哪些文件？
- 到底什么是端到端模型
- 什么叫长尾问题？怎么解决长尾分布问题？
- PPL (Perplexity) 及其数学公式
- 大海捞针测试和概率探针分别是什么

## Agent、MCP等技术。

## 核心组件代码实现

专门针对 “手撕” Transformer 关键组件的编程实现题，聚焦工程能力考察。

- 手撕自注意力机制(Self Attention)
- 手撕多头注意力机制(Multi Head Attention)
- 手撕 MQA
- 手撕绝对位置编码
- 手撕 RoPE
- 手撕 Transformer 中 FFN 代码
- 手撕 Layer Norm
- 手撕 RMSNorm
- 手撕 flash-attention
