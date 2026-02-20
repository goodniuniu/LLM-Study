"""
PyTorch基础教程 - Tensor操作
本教程介绍PyTorch中最基础的数据结构: Tensor(张量)
"""

import torch
import numpy as np


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


# ============================================================
# 1. Tensor创建
# ============================================================
print_section("1. Tensor创建")

# 1.1 从数据创建
print("【1.1 从数据创建】")
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(f"从列表创建: \n{x_data}")
print(f"数据类型: {x_data.dtype}")
print(f"设备: {x_data.device}")

# 从NumPy数组创建
np_array = np.array([[1, 2], [3, 4]])
x_np = torch.from_numpy(np_array)
print(f"\n从NumPy创建: \n{x_np}")

# 1.2 使用内置函数创建
print("\n【1.2 使用内置函数创建】")

# 全0张量
zeros = torch.zeros(3, 4)
print(f"全0张量 (3x4): \n{zeros}")

# 全1张量
ones = torch.ones(2, 3)
print(f"\n全1张量 (2x3): \n{ones}")

# 随机张量(均匀分布)
rand = torch.rand(2, 3)
print(f"\n随机张量 [0,1): \n{rand}")

# 随机张量(标准正态分布)
randn = torch.randn(2, 3)
print(f"\n正态分布随机张量: \n{randn}")

# 等差数列
arange = torch.arange(0, 10, 2)  # 从0到10,步长2
print(f"\n等差数列: {arange}")

# 线性空间
linspace = torch.linspace(0, 1, 5)  # 0到1之间5个数
print(f"线性空间: {linspace}")

# 1.3 创建与现有张量相同属性的张量
print("\n【1.3 继承属性的张量】")
x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
x_ones = torch.ones_like(x)
print(f"原张量: \n{x}")
print(f"相同形状的全1张量: \n{x_ones}")

x_rand = torch.rand_like(x)
print(f"\n相同形状的随机张量: \n{x_rand}")


# ============================================================
# 2. Tensor属性
# ============================================================
print_section("2. Tensor属性")

tensor = torch.randn(3, 4, 5)
print(f"张量: \n{tensor}")
print(f"\n形状 (shape): {tensor.shape}")
print(f"数据类型 (dtype): {tensor.dtype}")
print(f"设备 (device): {tensor.device}")
print(f"维度数 (ndim): {tensor.ndim}")
print(f"元素总数 (numel): {tensor.numel()}")


# ============================================================
# 3. Tensor运算
# ============================================================
print_section("3. Tensor运算")

# 创建两个张量
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
print(f"张量 A: \n{a}")
print(f"\n张量 B: \n{b}")

# 3.1 算术运算
print("\n【3.1 算术运算】")
print(f"加法 (a + b): \n{a + b}")
print(f"\n减法 (a - b): \n{a - b}")
print(f"\n乘法 (元素级 a * b): \n{a * b}")
print(f"\n除法 (a / b): \n{a / b}")
print(f"\n幂运算 (a ** 2): \n{a ** 2}")

# 3.2 矩阵运算
print("\n【3.2 矩阵运算】")
# 矩阵乘法
matmul = torch.matmul(a, b)
print(f"矩阵乘法 (a @ b): \n{matmul}")

# 或使用mm (2D张量)
mm = torch.mm(a, b)
print(f"\n矩阵乘法 (torch.mm): \n{mm}")

# 转置
transpose = a.T
print(f"\n转置 (a.T): \n{transpose}")

# 3.3 聚合运算
print("\n【3.3 聚合运算】")
print(f"求和 (sum): {a.sum()}")
print(f"均值 (mean): {a.mean()}")
print(f"最大值 (max): {a.max()}")
print(f"最小值 (min): {a.min()}")

# 按维度聚合
print(f"\n按行求和 (dim=0): {a.sum(dim=0)}")
print(f"按列求和 (dim=1): {a.sum(dim=1)}")

# 3.4 其他常用运算
print("\n【3.4 其他运算】")
print(f"绝对值: \n{torch.abs(torch.tensor([-1, -2, 3]))}")
print(f"\n指数: \n{torch.exp(a)}")
print(f"\n对数: \n{torch.log(torch.abs(a) + 1)}")


# ============================================================
# 4. Tensor索引和切片
# ============================================================
print_section("4. Tensor索引和切片")

tensor = torch.arange(24).reshape(4, 6)
print(f"原张量 (4x6): \n{tensor}")

# 4.1 索引
print("\n【4.1 索引】")
print(f"第0行: {tensor[0]}")
print(f"第0列: {tensor[:, 0]}")
print(f"第2行第3列: {tensor[2, 3]}")

# 4.2 切片
print("\n【4.2 切片】")
print(f"前2行: \n{tensor[:2]}")
print(f"\n后2列: \n{tensor[:, -2:]}")
print(f"\n子矩阵 [1:3, 2:5]: \n{tensor[1:3, 2:5]}")

# 4.3 高级索引
print("\n【4.3 高级索引】")
# 使用索引张量
indices = torch.tensor([0, 2])
print(f"第0和第2行: \n{tensor[indices]}")

# 布尔索引
mask = tensor > 15
print(f"\n大于15的元素: {tensor[mask]}")


# ============================================================
# 5. Tensor形状操作
# ============================================================
print_section("5. Tensor形状操作")

tensor = torch.arange(24)
print(f"原张量: {tensor.shape}")
print(tensor)

# 5.1 reshape和view
print("\n【5.1 reshape/view】")
reshaped = tensor.reshape(4, 6)
print(f"reshape为 (4, 6): {reshaped.shape}")

viewed = tensor.view(2, 3, 4)
print(f"view为 (2, 3, 4): {viewed.shape}")

# 5.2 squeeze和unsqueeze
print("\n【5.2 squeeze/unsqueeze】")
tensor_2d = torch.randn(3, 1, 4, 1)
print(f"原形状: {tensor_2d.shape}")

squeezed = tensor_2d.squeeze()
print(f"squeeze后: {squeezed.shape}")

unsqueezed = tensor.unsqueeze(0)
print(f"unsqueeze(0)后: {unsqueezed.shape}")

# 5.3 flatten和permute
print("\n【5.3 flatten/permute】")
tensor_3d = torch.randn(2, 3, 4)
print(f"原张量 (2,3,4): {tensor_3d.shape}")

flattened = tensor_3d.flatten()
print(f"flatten后: {flattened.shape}")

permuted = tensor_3d.permute(2, 0, 1)  # 交换维度
print(f"permute(2,0,1)后: {permuted.shape}")


# ============================================================
# 6. GPU Tensor操作
# ============================================================
print_section("6. GPU Tensor操作")

# 检查GPU是否可用
if torch.cuda.is_available():
    print(f"GPU可用: {torch.cuda.get_device_name(0)}")
    
    # 创建CPU张量
    cpu_tensor = torch.randn(3, 3)
    print(f"\nCPU张量设备: {cpu_tensor.device}")
    
    # 移动到GPU
    gpu_tensor = cpu_tensor.cuda()
    print(f"GPU张量设备: {gpu_tensor.device}")
    
    # 或者使用to方法
    gpu_tensor2 = cpu_tensor.to('cuda')
    print(f"GPU张量设备 (to方法): {gpu_tensor2.device}")
    
    # 移回CPU
    back_to_cpu = gpu_tensor.cpu()
    print(f"回到CPU: {back_to_cpu.device}")
    
    # 在GPU上创建张量
    direct_gpu = torch.randn(3, 3, device='cuda')
    print(f"\n直接在GPU创建: {direct_gpu.device}")
    
    # GPU运算
    a_gpu = torch.randn(1000, 1000, device='cuda')
    b_gpu = torch.randn(1000, 1000, device='cuda')
    
    import time
    start = time.time()
    c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"\nGPU矩阵乘法时间: {gpu_time:.4f}秒")
    
else:
    print("GPU不可用,跳过GPU相关操作")


# ============================================================
# 7. Tensor与NumPy转换
# ============================================================
print_section("7. Tensor与NumPy转换")

# Tensor转NumPy
tensor = torch.ones(5)
print(f"Tensor: {tensor}")
numpy_array = tensor.numpy()
print(f"NumPy数组: {numpy_array}")

# NumPy转Tensor
numpy_array = np.array([1, 2, 3])
tensor = torch.from_numpy(numpy_array)
print(f"\nNumPy数组: {numpy_array}")
print(f"Tensor: {tensor}")

# 注意: 它们共享内存
numpy_array[0] = 100
print(f"\n修改NumPy后:")
print(f"NumPy数组: {numpy_array}")
print(f"Tensor: {tensor}")


# ============================================================
# 8. 原地操作
# ============================================================
print_section("8. 原地操作")

tensor = torch.ones(3, 3)
print(f"原张量: \n{tensor}")

# 原地加法 (带_后缀)
tensor.add_(5)
print(f"\nadd_(5)后: \n{tensor}")

# 其他原地操作
tensor.mul_(2)
print(f"mul_(2)后: \n{tensor}")

print("\n注意: 原地操作会改变原张量的值,不会创建新张量")


print("\n" + "=" * 60)
print("  Tensor基础教程完成!")
print("=" * 60)
