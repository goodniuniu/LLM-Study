"""
PyTorch基础教程 - 自动求导(Autograd)
本教程介绍PyTorch的自动求导系统,这是神经网络训练的核心
"""

import torch
import torch.nn as nn


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


# ============================================================
# 1. 计算图和梯度基础
# ============================================================
print_section("1. 计算图和梯度基础")

print("PyTorch的autograd系统可以自动计算梯度")
print("只需要设置 requires_grad=True,PyTorch就会跟踪所有操作")

# 创建一个需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)
print(f"x = {x}")
print(f"requires_grad: {x.requires_grad}")

# 定义计算
y = x ** 2
print(f"\ny = x^2 = {y}")
print(f"y是计算的结果,它有grad_fn: {y.grad_fn}")

# 计算梯度
y.backward()
print(f"\n梯度 dy/dx = {x.grad}")
print("预期结果: dy/dx = 2x = 4.0")


# ============================================================
# 2. 多变量梯度计算
# ============================================================
print_section("2. 多变量梯度计算")

# 创建多个变量
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

print(f"x = {x}")
print(f"y = {y}")

# 定义函数: z = x^2 + y^3
z = x ** 2 + y ** 3
print(f"\nz = x^2 + y^3 = {z}")

# 计算梯度
z.backward()

print(f"\n梯度:")
print(f"  dz/dx = {x.grad}  (预期: 2x = 4)")
print(f"  dz/dy = {y.grad}  (预期: 3y^2 = 27)")


# ============================================================
# 3. 向量梯度计算
# ============================================================
print_section("3. 向量梯度计算")

# 创建向量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"向量 x = {x}")

# 定义函数: y = sum(x^2)
y = torch.sum(x ** 2)
print(f"\ny = sum(x^2) = {y}")

# 计算梯度
y.backward()

print(f"\n梯度 dy/dx = {x.grad}")
print("预期结果: [2, 4, 6]")


# ============================================================
# 4. 雅可比矩阵和向量-雅可比积
# ============================================================
print_section("4. 雅可比矩阵和向量-雅可比积")

print("当输出是向量时,backward需要传入梯度参数")

# 创建输入
x = torch.tensor([1.0, 2.0], requires_grad=True)
print(f"x = {x}")

# 定义向量函数
# y1 = x1^2 + 3*x2
# y2 = 2*x1 + x2^2
y = torch.stack([
    x[0] ** 2 + 3 * x[1],
    2 * x[0] + x[1] ** 2
])
print(f"\ny = [x1^2 + 3*x2, 2*x1 + x2^2] = {y}")

# 计算梯度(传入权重向量)
v = torch.tensor([1.0, 1.0])  # 权重向量
y.backward(v)

print(f"\n梯度:")
print(f"  dy/dx (带权重v={v}) = {x.grad}")
print("\n这计算的是: v^T * J, 其中J是雅可比矩阵")


# ============================================================
# 5. 阻止梯度跟踪
# ============================================================
print_section("5. 阻止梯度跟踪")

x = torch.tensor(2.0, requires_grad=True)
print(f"x = {x}, requires_grad = {x.requires_grad}")

# 方法1: detach()
y = x ** 2
z = y.detach()
print(f"\ny = x^2 = {y}")
print(f"z = y.detach() = {z}")
print(f"z.requires_grad = {z.requires_grad}")

# 方法2: torch.no_grad()
with torch.no_grad():
    w = x ** 2
    print(f"\n在no_grad中: w = x^2 = {w}")
    print(f"w.requires_grad = {w.requires_grad}")

# 方法3: 设置requires_grad=False
x_no_grad = x.clone().detach()
print(f"\nclone().detach()后: requires_grad = {x_no_grad.requires_grad}")


# ============================================================
# 6. 梯度累积和清零
# ============================================================
print_section("6. 梯度累积和清零")

x = torch.tensor(2.0, requires_grad=True)

print("注意: PyTorch默认会累积梯度!")
print(f"初始x.grad = {x.grad}")

# 第一次反向传播
y1 = x ** 2
y1.backward()
print(f"\n第一次: y = x^2")
print(f"x.grad = {x.grad} (预期: 4)")

# 第二次反向传播(不清零)
y2 = x ** 3
y2.backward()
print(f"\n第二次: y = x^3 (未清零梯度)")
print(f"x.grad = {x.grad} (累积了: 4 + 12 = 16)")

# 清零梯度
x.grad.zero_()
print(f"\n清零后: x.grad = {x.grad}")

# 重新计算
y3 = x ** 2
y3.backward()
print(f"\n重新计算 y = x^2")
print(f"x.grad = {x.grad} (正确: 4)")


# ============================================================
# 7. 复杂计算图示例
# ============================================================
print_section("7. 复杂计算图示例")

# 定义参数
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)  # 输入(不需要梯度)

print(f"参数: w = {w}, b = {b}")
print(f"输入: x = {x}")

# 前向传播
u = w * x
v = u + b
y = v ** 2

print(f"\n计算过程:")
print(f"  u = w * x = {u}")
print(f"  v = u + b = {v}")
print(f"  y = v^2 = {y}")

# 反向传播
y.backward()

print(f"\n梯度:")
print(f"  dy/dw = {w.grad}")
print(f"  dy/db = {b.grad}")

# 手动验证
# y = (w*x + b)^2
# dy/dw = 2*(w*x + b)*x = 2*7*3 = 42
# dy/db = 2*(w*x + b) = 2*7 = 14
print("\n手动验证:")
print("  dy/dw = 2*(w*x+b)*x = 2*7*3 = 42")
print("  dy/db = 2*(w*x+b) = 2*7 = 14")


# ============================================================
# 8. 神经网络中的梯度计算
# ============================================================
print_section("8. 神经网络中的梯度计算")

# 定义一个简单的线性层
linear = nn.Linear(3, 2)
print(f"线性层: {linear}")
print(f"权重形状: {linear.weight.shape}")
print(f"偏置形状: {linear.bias.shape}")

# 输入
x = torch.randn(1, 3)
print(f"\n输入 x: {x}")

# 前向传播
output = linear(x)
print(f"输出: {output}")

# 定义损失
target = torch.tensor([[1.0, 0.0]])
loss = nn.MSELoss()(output, target)
print(f"\n目标: {target}")
print(f"损失 (MSE): {loss.item():.4f}")

# 反向传播
loss.backward()

print(f"\n梯度:")
print(f"  权重梯度形状: {linear.weight.grad.shape}")
print(f"  权重梯度: \n{linear.weight.grad}")
print(f"  偏置梯度: {linear.bias.grad}")


# ============================================================
# 9. 自定义函数(高级)
# ============================================================
print_section("9. 自定义函数")

class MySquare(torch.autograd.Function):
    """
    自定义函数: f(x) = x^2
    需要实现forward和backward
    """
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = 2 * input * grad_output
        return grad_input


# 使用自定义函数
x = torch.tensor(3.0, requires_grad=True)
my_square = MySquare.apply

y = my_square(x)
print(f"x = {x}")
print(f"y = MySquare(x) = {y}")

y.backward()
print(f"\n梯度 dy/dx = {x.grad}")
print("预期结果: 2*x = 6")


# ============================================================
# 10. 实际应用: 简单的线性回归
# ============================================================
print_section("10. 实际应用: 简单线性回归")

# 生成数据
# 真实参数: y = 2x + 1
torch.manual_seed(42)
x = torch.randn(100, 1)
y_true = 2 * x + 1 + 0.1 * torch.randn(100, 1)  # 添加噪声

print(f"数据形状: x={x.shape}, y={y_true.shape}")

# 初始化参数
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"\n初始参数: w={w.item():.4f}, b={b.item():.4f}")

# 训练
learning_rate = 0.1
for epoch in range(100):
    # 前向传播
    y_pred = w * x + b
    
    # 计算损失
    loss = torch.mean((y_pred - y_true) ** 2)
    
    # 反向传播
    loss.backward()
    
    # 更新参数(不使用优化器,手动更新)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # 清零梯度
        w.grad.zero_()
        b.grad.zero_()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\n最终参数: w={w.item():.4f}, b={b.item():.4f}")
print("真实参数: w=2.0, b=1.0")


print("\n" + "=" * 60)
print("  自动求导教程完成!")
print("=" * 60)
print("\n关键要点:")
print("1. 设置 requires_grad=True 来跟踪梯度")
print("2. 调用 .backward() 计算梯度")
print("3. 梯度存储在 .grad 属性中")
print("4. 每次反向传播前需要清零梯度")
print("5. 使用 torch.no_grad() 或 detach() 阻止梯度跟踪")
