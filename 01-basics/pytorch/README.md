# PyTorch 框架基础

学习PyTorch深度学习框架的使用。

## 学习目标
- 掌握Tensor操作
- 理解自动求导机制
- 熟悉神经网络构建
- 掌握模型训练流程

## 学习路径

### 1. Tensor基础 (`01_tensor_basics.py`)
- Tensor创建和属性
- Tensor运算和操作
- Tensor索引和切片
- GPU Tensor操作

### 2. 自动求导 (`02_autograd.py`)
- 计算图概念
- 梯度计算
- 反向传播
- 梯度清零

### 3. 神经网络基础 (`03_neural_network.py`)
- 使用nn.Module
- 激活函数
- 损失函数
- 优化器

### 4. 数据加载 (`04_data_loading.py`)
- Dataset类
- DataLoader
- 数据变换
- 批处理

### 5. 完整训练流程 (`05_training_pipeline.py`)
- 模型定义
- 训练循环
- 验证和测试
- 模型保存和加载

### 6. GPU加速 (`06_gpu_acceleration.py`)
- 设备管理
- 模型迁移
- 混合精度训练

## 运行代码

```bash
# 确保虚拟环境已激活
activate.bat

# 运行示例
cd 01-basics/pytorch
python 01_tensor_basics.py
```

## 参考资料
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [PyTorch文档](https://pytorch.org/docs/stable/index.html)
- Deep Learning with PyTorch
