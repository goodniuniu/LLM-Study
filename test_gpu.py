#!/usr/bin/env python3
"""
GPU测试脚本 - 验证PyTorch是否能正常使用GPU
"""

import torch
import time
import sys


def print_separator(char="=", length=60):
    """打印分隔线"""
    print(char * length)


def check_cuda_available():
    """检查CUDA是否可用"""
    print("\n[1] CUDA可用性检查")
    print_separator("-", 40)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {cuda_available}")
    
    if not cuda_available:
        print("错误: CUDA不可用!")
        print("可能原因:")
        print("  - 未安装NVIDIA驱动")
        print("  - 安装的PyTorch是CPU版本")
        print("  - CUDA版本不匹配")
        return False
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    return True


def check_gpu_info():
    """检查GPU信息"""
    print("\n[2] GPU信息")
    print_separator("-", 40)
    
    gpu_count = torch.cuda.device_count()
    print(f"GPU数量: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"  计算能力: {props.major}.{props.minor}")
        print(f"  多处理器数量: {props.multi_processor_count}")


def test_tensor_operations():
    """测试张量操作"""
    print("\n[3] GPU张量操作测试")
    print_separator("-", 40)
    
    # 创建CPU张量
    print("创建CPU张量...")
    cpu_tensor = torch.randn(1000, 1000)
    print(f"  CPU张量大小: {cpu_tensor.shape}")
    print(f"  CPU张量设备: {cpu_tensor.device}")
    
    # 移动到GPU
    print("\n移动到GPU...")
    gpu_tensor = cpu_tensor.cuda()
    print(f"  GPU张量设备: {gpu_tensor.device}")
    
    # GPU运算
    print("\n执行GPU矩阵乘法...")
    start_time = time.time()
    result = torch.mm(gpu_tensor, gpu_tensor)
    torch.cuda.synchronize()  # 等待GPU完成
    gpu_time = time.time() - start_time
    print(f"  GPU计算时间: {gpu_time:.4f} 秒")
    print(f"  结果张量大小: {result.shape}")
    
    # CPU运算对比
    print("\n执行CPU矩阵乘法(对比)...")
    start_time = time.time()
    result_cpu = torch.mm(cpu_tensor, cpu_tensor)
    cpu_time = time.time() - start_time
    print(f"  CPU计算时间: {cpu_time:.4f} 秒")
    
    # 计算加速比
    if cpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"\n  GPU加速比: {speedup:.2f}x")


def test_neural_network():
    """测试神经网络训练"""
    print("\n[4] 神经网络测试")
    print_separator("-", 40)
    
    # 创建一个简单的神经网络
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10)
    )
    
    print(f"模型结构:\n{model}")
    
    # 移动模型到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"\n模型已移动到: {device}")
    
    # 创建模拟数据
    batch_size = 64
    input_data = torch.randn(batch_size, 1000).to(device)
    target = torch.randint(0, 10, (batch_size,)).to(device)
    
    print(f"\n输入数据大小: {input_data.shape}")
    print(f"目标大小: {target.shape}")
    
    # 前向传播
    print("\n执行前向传播...")
    output = model(input_data)
    print(f"输出大小: {output.shape}")
    
    # 计算损失
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(output, target)
    print(f"损失值: {loss.item():.4f}")
    
    # 反向传播
    print("执行反向传播...")
    loss.backward()
    print("反向传播完成!")
    
    # 检查显存使用
    print(f"\n显存使用情况:")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"  保留: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


def test_training_loop():
    """测试训练循环"""
    print("\n[5] 训练循环测试")
    print_separator("-", 40)
    
    device = torch.device("cuda")
    
    # 简单模型
    model = torch.nn.Linear(100, 10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # 模拟训练
    print("执行10轮训练...")
    for epoch in range(10):
        # 生成随机数据
        inputs = torch.randn(32, 100).to(device)
        targets = torch.randn(32, 10).to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
    
    print("训练循环测试完成!")


def benchmark_performance():
    """性能基准测试"""
    print("\n[6] 性能基准测试")
    print_separator("-", 40)
    
    sizes = [1000, 2000, 4000]
    
    for size in sizes:
        print(f"\n矩阵大小: {size}x{size}")
        
        # CPU测试
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        
        start = time.time()
        c_cpu = torch.mm(a_cpu, b_cpu)
        cpu_time = time.time() - start
        print(f"  CPU时间: {cpu_time:.4f} 秒")
        
        # GPU测试
        if torch.cuda.is_available():
            a_gpu = a_cpu.cuda()
            b_gpu = b_cpu.cuda()
            
            # 预热
            torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            
            start = time.time()
            c_gpu = torch.mm(a_gpu, b_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            print(f"  GPU时间: {gpu_time:.4f} 秒")
            
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                print(f"  加速比: {speedup:.2f}x")


def main():
    """主函数"""
    print_separator("=")
    print("PyTorch GPU测试脚本")
    print_separator("=")
    print(f"Python版本: {sys.version}")
    
    # 检查CUDA
    if not check_cuda_available():
        print("\n测试中止: CUDA不可用")
        return
    
    # 检查GPU信息
    check_gpu_info()
    
    # 测试张量操作
    test_tensor_operations()
    
    # 测试神经网络
    test_neural_network()
    
    # 测试训练循环
    test_training_loop()
    
    # 性能基准测试
    benchmark_performance()
    
    print("\n")
    print_separator("=")
    print("所有测试完成!")
    print_separator("=")
    print("\n结论: GPU可以正常工作,可以开始深度学习训练!")


if __name__ == "__main__":
    main()
