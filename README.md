# LLM-Study 深度学习学习项目

本项目用于系统学习深度学习和大型语言模型(LLM)相关知识。

## 项目结构

```
LLM-Study/
├── 01-basics/              # 基础知识
│   ├── python-basics/      # Python基础
│   ├── numpy/              # NumPy数组操作
│   ├── pytorch/            # PyTorch框架基础
│   └── ml-basics/          # 机器学习基础
│
├── 02-neural-networks/     # 神经网络
│   ├── mlp/               # 多层感知机
│   ├── cnn/               # 卷积神经网络
│   ├── rnn/               # 循环神经网络
│   └── attention/         # 注意力机制
│
├── 03-transformer/        # Transformer架构
│   ├── attention-mechanism/    # 注意力机制详解
│   ├── transformer-architecture/ # Transformer架构
│   ├── bert/                    # BERT模型
│   └── gpt/                     # GPT系列
│
├── 04-llm-foundation/    # LLM基础
│   ├── pre-training/     # 预训练
│   ├── fine-tuning/      # 微调技术
│   ├── prompt-eng/       # 提示工程
│   └── scaling-laws/     # 扩展定律
│
├── 05-training/          # 训练技术
│   ├── data-preprocessing/   # 数据预处理
│   ├── optimization/         # 优化算法
│   ├── distributed-training/ # 分布式训练
│   └── rlhf/                 # RLHF训练
│
├── 06-deployment/        # 部署与推理
│   ├── model-quantization/   # 模型量化
│   ├── inference-optimization/ # 推理优化
│   ├── api-deployment/       # API部署
│   └── edge-deployment/      # 边缘部署
│
├── 07-practice/          # 实战项目
│   ├── text-classification/    # 文本分类
│   ├── text-generation/        # 文本生成
│   ├── question-answering/     # 问答系统
│   └── code-generation/        # 代码生成
│
├── 08-advanced/          # 进阶主题
│   ├── multimodal/       # 多模态模型
│   ├── agents/           # AI Agent
│   └── rag/              # 检索增强生成
│
├── data/                 # 数据集
│   ├── raw/             # 原始数据
│   ├── processed/       # 处理后数据
│   └── cache/           # 缓存数据
│
├── models/               # 模型文件
│   ├── checkpoints/      # 模型检查点
│   ├── pretrained/       # 预训练模型
│   └── exported/         # 导出模型
│
├── notebooks/            # Jupyter笔记本
│
├── scripts/              # 工具脚本
│   ├── data/            # 数据处理脚本
│   ├── train/           # 训练脚本
│   └── utils/           # 工具函数
│
├── docs/                 # 学习笔记
│
├── requirements.txt     # Python依赖
└── .gitignore           # Git忽略文件
```

## 学习路径

1. **基础篇** (01-basics): Python、PyTorch、机器学习基础
2. **神经网络篇** (02-neural-networks): 传统神经网络架构
3. **Transformer篇** (03-transformer): Transformer及其变体
4. **LLM基础篇** (04-llm-foundation): LLM核心技术
5. **训练篇** (05-training): 高效训练技术
6. **部署篇** (06-deployment): 模型部署与优化
7. **实战篇** (07-practice): 实际应用项目
8. **进阶篇** (08-advanced): 前沿技术探索

## 环境配置

```bash
pip install -r requirements.txt
```

## 快速开始

每个目录都包含对应的学习内容和代码示例。建议按照数字顺序逐步学习。
