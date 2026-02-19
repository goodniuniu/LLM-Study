# LLM-Study 深度学习学习项目

本项目用于系统学习深度学习和大型语言模型(LLM)相关知识。

## 项目特点

- 循序渐进的学习路径，从基础到进阶
- 完整的PyTorch教程，包含代码示例
- 虚拟环境管理，依赖清晰
- GPU加速支持
- Git版本控制，记录学习进度

## 项目结构

```
LLM-Study/
├── 01-basics/              # 基础知识
│   ├── python-basics/      # Python基础
│   ├── numpy/              # NumPy数组操作
│   ├── pytorch/            # PyTorch框架基础(6个教程)
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
├── venv/                 # 虚拟环境(自动创建)
├── requirements.txt      # Python依赖
├── setup_env.bat         # 环境安装脚本
├── activate.bat          # 快速激活脚本
├── check_env.py          # 环境检查脚本
├── test_gpu.py           # GPU测试脚本
├── START.md              # 快速开始指南
├── GIT_GUIDE.md          # Git使用指南
└── .gitignore            # Git忽略文件
```

## 使用方法

### 1. 首次使用 - 环境配置

#### 方式一: 自动安装(推荐)

```bash
# 双击运行自动安装脚本
setup_env.bat
```

脚本会自动:
1. 创建虚拟环境 `venv/`
2. 激活虚拟环境
3. 安装所有依赖包
4. 配置GPU支持(如果可用)

#### 方式二: 手动安装

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
# Windows:
venv\Scripts\activate
# 或使用快捷脚本:
activate.bat

# 3. 升级pip
python -m pip install --upgrade pip

# 4. 安装依赖
pip install -r requirements.txt

# 5. 安装GPU版PyTorch(可选,需要NVIDIA显卡)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. 验证环境

```bash
# 检查环境是否完备
python check_env.py

# 测试GPU是否正常工作(如果有NVIDIA显卡)
python test_gpu.py
```

### 3. 开始学习

#### 学习路径

建议按照以下顺序学习:

1. **基础篇** (01-basics)
   - Python基础 → NumPy → PyTorch → 机器学习基础

2. **神经网络篇** (02-neural-networks)
   - MLP → CNN → RNN → Attention

3. **Transformer篇** (03-transformer)
   - Attention机制 → Transformer架构 → BERT → GPT

4. **LLM基础篇** (04-llm-foundation)
   - 预训练 → 微调 → 提示工程 → 扩展定律

5. **训练篇** (05-training)
   - 数据预处理 → 优化算法 → 分布式训练 → RLHF

6. **部署篇** (06-deployment)
   - 模型量化 → 推理优化 → API部署 → 边缘部署

7. **实战篇** (07-practice)
   - 文本分类 → 文本生成 → 问答系统 → 代码生成

8. **进阶篇** (08-advanced)
   - 多模态模型 → AI Agent → RAG

#### 运行教程

以PyTorch基础教程为例:

```bash
# 1. 确保虚拟环境已激活
activate.bat

# 2. 进入教程目录
cd 01-basics\pytorch

# 3. 按顺序运行教程
python 01_tensor_basics.py
python 02_autograd.py
python 03_neural_network.py
python 04_data_loading.py
python 05_training_pipeline.py
python 06_gpu_acceleration.py
```

每个教程都是独立可运行的Python脚本，包含:
- 详细的代码注释
- 实际运行示例
- 关键概念解释

### 4. 记录学习进度(Git)

项目已配置Git版本控制，建议定期提交学习进度:

```bash
# 查看当前状态
git status

# 添加修改的文件
git add 01-basics/pytorch/01_tensor_basics.py

# 提交(带详细说明)
git commit -m "feat: 完成Tensor基础教程学习

- 学习Tensor创建和操作
- 理解Tensor属性和运算  
- 掌握GPU Tensor操作"

# 查看提交历史
git log --oneline
```

详细的Git使用指南请参考 `GIT_GUIDE.md`

### 5. 日常使用流程

```bash
# 1. 进入项目目录
cd c:\Users\admin\GitHub\LLM-Study

# 2. 激活虚拟环境
activate.bat

# 3. 检查环境
python check_env.py

# 4. 开始学习
# 例如: 运行PyTorch教程
cd 01-basics\pytorch
python 01_tensor_basics.py

# 5. 学习完成后提交进度
cd c:\Users\admin\GitHub\LLM-Study
git add .
git commit -m "feat: 完成今日学习"
```

## 学习资源

### PyTorch教程(已包含在项目中)

| 教程 | 内容 | 文件 |
|------|------|------|
| Tensor基础 | Tensor创建、操作、索引、GPU使用 | `01_tensor_basics.py` |
| 自动求导 | 计算图、梯度计算、反向传播 | `02_autograd.py` |
| 神经网络 | nn.Module、层类型、激活函数、损失函数 | `03_neural_network.py` |
| 数据加载 | Dataset、DataLoader、Transforms | `04_data_loading.py` |
| 训练流程 | 完整训练、验证、测试、保存模型 | `05_training_pipeline.py` |
| GPU加速 | GPU管理、性能优化、混合精度 | `06_gpu_acceleration.py` |

### 外部资源

- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [PyTorch文档](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [深度学习花书](https://www.deeplearningbook.org/)

## 常见问题

### Q: 如何切换CPU/GPU版本PyTorch?

```bash
# 卸载当前版本
pip uninstall torch torchvision torchaudio

# 安装CPU版本
pip install torch torchvision torchaudio

# 安装GPU版本(CUDA 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Q: pip下载速度慢怎么办?

```bash
# 使用清华镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名

# 或永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 虚拟环境如何删除重建?

```bash
# 1. 退出虚拟环境
deactivate

# 2. 删除虚拟环境文件夹
rmdir /s venv

# 3. 重新创建
python -m venv venv
```

### Q: 如何查看已安装的包?

```bash
pip list

# 或查看特定包
pip show torch
```

## 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8 或更高版本
- **内存**: 建议 8GB 以上
- **显卡**: NVIDIA显卡(可选,用于GPU加速)
  - 需要CUDA支持
  - 推荐显存 4GB 以上

## 贡献与反馈

如果你发现任何问题或有改进建议,欢迎:
- 提交Issue
- 提交Pull Request
- 分享学习心得

## 许可证

本项目仅供学习使用。

---

**祝你学习愉快!** 🚀

如有问题,请参考:
- `START.md` - 快速开始指南
- `GIT_GUIDE.md` - Git使用指南
- 各教程文件中的详细注释
