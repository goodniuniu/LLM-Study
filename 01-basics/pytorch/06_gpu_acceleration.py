"""
PyTorch基础教程 - GPU加速
本教程介绍如何使用GPU加速深度学习训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


# ============================================================
# 1. GPU可用性检查
# ============================================================
print_section("1. GPU可用性检查")

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")
        print(f"  多处理器数量: {props.multi_processor_count}")
else:
    print("\n警告: GPU不可用,将使用CPU运行")


# ============================================================
# 2. 设备管理
# ============================================================
print_section("2. 设备管理")

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前设备: {device}")

# 创建张量并移动到设备
print("\n【2.1 创建GPU张量】")

# 方式1: 在CPU创建后移动到GPU
x_cpu = torch.randn(3, 3)
print(f"CPU张量设备: {x_cpu.device}")

if torch.cuda.is_available():
    x_gpu = x_cpu.cuda()
    print(f"移动到GPU后: {x_gpu.device}")
    
    # 方式2: 使用to方法
    x_gpu2 = x_cpu.to('cuda')
    print(f"使用to方法: {x_gpu2.device}")
    
    # 方式3: 直接在GPU创建
    x_direct = torch.randn(3, 3, device='cuda')
    print(f"直接在GPU创建: {x_direct.device}")
    
    # 方式4: 使用设备对象
    x_device = torch.randn(3, 3, device=device)
    print(f"使用device对象: {x_device.device}")

print("\n【2.2 张量移回CPU】")
if torch.cuda.is_available():
    x_back = x_gpu.cpu()
    print(f"移回CPU: {x_back.device}")
    
    # 转换为numpy(必须先移回CPU)
    x_numpy = x_gpu.cpu().numpy()
    print(f"转换为NumPy数组: 形状={x_numpy.shape}")


# ============================================================
# 3. 模型迁移到GPU
# ============================================================
print_section("3. 模型迁移到GPU")


class SimpleNet(nn.Module):
    """简单的神经网络"""
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 创建模型
model = SimpleNet()
print(f"模型创建后设备: {next(model.parameters()).device}")

# 移动模型到GPU
if torch.cuda.is_available():
    model = model.to(device)
    print(f"移动到GPU后: {next(model.parameters()).device}")

print("\n注意: 模型和数据必须在同一设备上!")


# ============================================================
# 4. CPU vs GPU性能对比
# ============================================================
print_section("4. CPU vs GPU性能对比")


def benchmark_matrix_multiply(size, device, iterations=100):
    """基准测试: 矩阵乘法"""
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 预热
    torch.mm(a, b)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    start = time.time()
    for _ in range(iterations):
        c = torch.mm(a, b)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed


def benchmark_neural_network(batch_size, device, iterations=50):
    """基准测试: 神经网络前向传播"""
    model = SimpleNet().to(device)
    x = torch.randn(batch_size, 784, device=device)
    
    # 预热
    model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            output = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    elapsed = time.time() - start
    
    return elapsed


print("【4.1 矩阵乘法性能对比】")
sizes = [1000, 2000, 4000]

for size in sizes:
    print(f"\n矩阵大小: {size}x{size}")
    
    # CPU测试
    cpu_time = benchmark_matrix_multiply(size, torch.device('cpu'))
    print(f"  CPU时间: {cpu_time:.4f}秒")
    
    # GPU测试
    if torch.cuda.is_available():
        gpu_time = benchmark_matrix_multiply(size, torch.device('cuda'))
        print(f"  GPU时间: {gpu_time:.4f}秒")
        speedup = cpu_time / gpu_time
        print(f"  GPU加速比: {speedup:.2f}x")

print("\n【4.2 神经网络性能对比】")
batch_sizes = [64, 256, 1024]

for batch_size in batch_sizes:
    print(f"\n批次大小: {batch_size}")
    
    # CPU测试
    cpu_time = benchmark_neural_network(batch_size, torch.device('cpu'))
    print(f"  CPU时间: {cpu_time:.4f}秒")
    
    # GPU测试
    if torch.cuda.is_available():
        gpu_time = benchmark_neural_network(batch_size, torch.device('cuda'))
        print(f"  GPU时间: {gpu_time:.4f}秒")
        speedup = cpu_time / gpu_time
        print(f"  GPU加速比: {speedup:.2f}x")


# ============================================================
# 5. GPU内存管理
# ============================================================
print_section("5. GPU内存管理")

if torch.cuda.is_available():
    print("【5.1 查看显存使用情况】")
    
    # 清理缓存
    torch.cuda.empty_cache()
    
    print(f"已分配显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"保留显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"最大分配显存: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    print("\n【5.2 显存分配演示】")
    
    # 创建大张量
    x = torch.randn(1000, 1000, device='cuda')
    print(f"创建张量后 - 已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # 创建更多张量
    y = torch.randn(1000, 1000, device='cuda')
    z = torch.randn(1000, 1000, device='cuda')
    print(f"创建3个张量后 - 已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # 删除张量
    del x, y, z
    print(f"删除张量后 - 已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # 清理缓存
    torch.cuda.empty_cache()
    print(f"清理缓存后 - 已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    print("\n【5.3 显存优化技巧】")
    print("  1. 及时删除不需要的张量: del tensor")
    print("  2. 清理缓存: torch.cuda.empty_cache()")
    print("  3. 使用torch.no_grad()减少显存占用")
    print("  4. 减小batch_size")
    print("  5. 使用梯度累积")
    print("  6. 使用混合精度训练")
else:
    print("GPU不可用,跳过显存管理演示")


# ============================================================
# 6. 多GPU训练
# ============================================================
print_section("6. 多GPU训练")

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"检测到 {torch.cuda.device_count()} 个GPU")
    print("\n可以使用DataParallel或DistributedDataParallel进行多GPU训练")
    
    # DataParallel示例
    model = SimpleNet()
    model = nn.DataParallel(model)
    model = model.to(device)
    print("\n使用DataParallel包装模型")
    print("  模型会自动分配到所有可用GPU")
    
else:
    print(f"GPU数量: {torch.cuda.device_count()}")
    print("多GPU训练需要至少2个GPU")


# ============================================================
# 7. 混合精度训练
# ============================================================
print_section("7. 混合精度训练 (Automatic Mixed Precision)")

if torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
    
    print("混合精度训练可以:")
    print("  - 减少显存占用")
    print("  - 加速训练(在支持Tensor Core的GPU上)")
    print("  - 保持模型精度")
    
    # 创建模型和数据
    model = SimpleNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 创建GradScaler
    scaler = GradScaler()
    
    # 模拟训练数据
    x = torch.randn(32, 784, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    
    print("\n【标准训练】")
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"损失: {loss.item():.4f}")
    
    print("\n【混合精度训练】")
    optimizer.zero_grad()
    
    # 使用autocast进行前向传播
    with autocast():
        output = model(x)
        loss = criterion(output, y)
    
    # 使用scaler进行反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"损失: {loss.item():.4f}")
    print("\n混合精度训练代码模板:")
    print("""
    from torch.cuda.amp import autocast, GradScaler
    
    scaler = GradScaler()
    
    for data, target in dataloader:
        optimizer.zero_grad()
        
        with autocast():  # 自动选择精度
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    """)
else:
    print("混合精度训练需要GPU支持")


# ============================================================
# 8. 完整的GPU训练示例
# ============================================================
print_section("8. 完整的GPU训练示例")


def train_with_gpu():
    """使用GPU进行完整训练的示例"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")
    
    # 创建数据
    X = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 创建模型并移到GPU
    model = SimpleNet().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # 将数据移到GPU
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"平均损失: {avg_loss:.4f}")
    
    return model


print("开始训练...")
model = train_with_gpu()
print("训练完成!")


# ============================================================
# 9. GPU训练最佳实践
# ============================================================
print_section("9. GPU训练最佳实践")

print("【9.1 设备管理】")
print("  1. 使用device对象统一管理设备")
print("  2. 模型和数据都要移到同一设备")
print("  3. 使用.to(device)而不是.cuda()")

print("\n【9.2 显存优化】")
print("  1. 使用torch.no_grad()进行推理")
print("  2. 及时删除不需要的中间结果")
print("  3. 使用混合精度训练")
print("  4. 使用梯度检查点节省显存")

print("\n【9.3 性能优化】")
print("  1. 使用pin_memory=True加速数据传输")
print("  2. 使用非阻塞数据传输: tensor.to(device, non_blocking=True)")
print("  3. 使用torch.backends.cudnn.benchmark = True")
print("  4. 在训练前调用torch.cuda.synchronize()进行预热")

print("\n【9.4 调试技巧】")
print("  1. 使用torch.cuda.memory_summary()查看显存使用")
print("  2. 使用torch.autograd.profiler分析性能")
print("  3. 监控GPU利用率(nvidia-smi)")

# 启用cuDNN自动调优
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("\n已启用cuDNN自动调优")


# ============================================================
# 10. 常见问题解决
# ============================================================
print_section("10. 常见问题解决")

print("【Q1: RuntimeError: Expected all tensors to be on the same device】")
print("  A: 确保模型和数据都在同一设备上")
print("     model = model.to(device)")
print("     data = data.to(device)")

print("\n【Q2: CUDA out of memory】")
print("  A: 1. 减小batch_size")
print("     2. 使用torch.cuda.empty_cache()")
print("     3. 使用混合精度训练")
print("     4. 使用梯度累积")

print("\n【Q3: GPU利用率低】")
print("  A: 1. 增加batch_size")
print("     2. 使用num_workers>0")
print("     3. 使用pin_memory=True")
print("     4. 检查数据加载是否成为瓶颈")

print("\n【Q4: 训练速度慢】")
print("  A: 1. 启用torch.backends.cudnn.benchmark")
print("     2. 使用混合精度训练")
print("     3. 使用多GPU训练")
print("     4. 优化数据预处理流程")


print("\n" + "=" * 60)
print("  GPU加速教程完成!")
print("=" * 60)
print("\n关键要点:")
print("1. 使用device对象统一管理设备")
print("2. 模型和数据必须在同一设备上")
print("3. GPU在大规模计算上有显著优势")
print("4. 注意显存管理,及时释放资源")
print("5. 使用混合精度训练加速并节省显存")
print("6. 监控GPU利用率优化性能")
