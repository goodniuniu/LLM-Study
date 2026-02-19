"""
PyTorch基础教程 - 神经网络构建
本教程介绍如何使用PyTorch构建神经网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


# ============================================================
# 1. 使用 nn.Module 构建网络
# ============================================================
print_section("1. 使用 nn.Module 构建网络")


class SimpleNet(nn.Module):
    """
    简单的神经网络示例
    结构: 输入层 -> 隐藏层 -> 输出层
    """
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        
        # 定义层
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """前向传播"""
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 创建网络实例
model = SimpleNet(input_size=784, hidden_size=256, num_classes=10)
print("网络结构:")
print(model)

# 查看模型参数
print(f"\n模型参数:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

# 测试前向传播
x = torch.randn(4, 784)  # 4个样本,每个784维
output = model(x)
print(f"\n输入形状: {x.shape}")
print(f"输出形状: {output.shape}")


# ============================================================
# 2. 使用 nn.Sequential 快速构建
# ============================================================
print_section("2. 使用 nn.Sequential 快速构建")

# 方式1: 直接传入层
model_seq = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

print("Sequential模型:")
print(model_seq)

# 方式2: 使用OrderedDict(可以给层命名)
from collections import OrderedDict

model_named = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 128)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(128, 10))
]))

print("\n带命名的Sequential模型:")
print(model_named)

# 测试
output = model_seq(x)
print(f"\n输出形状: {output.shape}")


# ============================================================
# 3. 常用层类型
# ============================================================
print_section("3. 常用层类型")

print("【3.1 全连接层 (Linear)】")
linear = nn.Linear(10, 5)
print(f"Linear(10, 5): 输入10维,输出5维")
x = torch.randn(2, 10)
output = linear(x)
print(f"输入: {x.shape} -> 输出: {output.shape}")

print("\n【3.2 卷积层 (Conv2d)】")
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
print(f"Conv2d(3, 16, 3, padding=1)")
x = torch.randn(2, 3, 32, 32)  # (batch, channels, height, width)
output = conv(x)
print(f"输入: {x.shape} -> 输出: {output.shape}")

print("\n【3.3 池化层 (MaxPool2d)】")
pool = nn.MaxPool2d(kernel_size=2, stride=2)
print(f"MaxPool2d(2, 2)")
output = pool(output)
print(f"输入: {x.shape} -> 输出: {output.shape}")

print("\n【3.4 批归一化 (BatchNorm)】")
bn = nn.BatchNorm2d(16)
print(f"BatchNorm2d(16)")
x = torch.randn(4, 16, 32, 32)
output = bn(x)
print(f"输入: {x.shape} -> 输出: {output.shape}")

print("\n【3.5 Dropout】")
dropout = nn.Dropout(p=0.5)
print(f"Dropout(p=0.5)")
x = torch.randn(2, 10)
output = dropout(x)
print(f"输入: {x.shape} -> 输出: {output.shape}")


# ============================================================
# 4. 激活函数
# ============================================================
print_section("4. 激活函数")

x = torch.linspace(-3, 3, 100)

print("常用激活函数:")
print("  - ReLU: max(0, x)")
print("  - Sigmoid: 1 / (1 + exp(-x))")
print("  - Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))")
print("  - Softmax: exp(x_i) / sum(exp(x_j))")
print("  - LeakyReLU: max(0, x) + negative_slope * min(0, x)")

# 使用示例
x = torch.tensor([-1.0, 0.0, 1.0])

print(f"\n输入: {x}")
print(f"ReLU: {F.relu(x)}")
print(f"Sigmoid: {torch.sigmoid(x)}")
print(f"Tanh: {torch.tanh(x)}")

# Softmax(用于多分类)
x = torch.tensor([1.0, 2.0, 3.0])
print(f"\nSoftmax输入: {x}")
print(f"Softmax: {F.softmax(x, dim=0)}")
print(f"Softmax之和: {F.softmax(x, dim=0).sum()}")


# ============================================================
# 5. 损失函数
# ============================================================
print_section("5. 损失函数")

# 创建预测和目标
predictions = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))

print("【5.1 交叉熵损失 (CrossEntropyLoss)】")
print("用于多分类问题")
criterion = nn.CrossEntropyLoss()
loss = criterion(predictions, targets)
print(f"预测: {predictions.shape}")
print(f"目标: {targets}")
print(f"损失值: {loss.item():.4f}")

print("\n【5.2 均方误差 (MSELoss)】")
print("用于回归问题")
pred = torch.randn(4, 5)
target = torch.randn(4, 5)
criterion = nn.MSELoss()
loss = criterion(pred, target)
print(f"预测: {pred.shape}")
print(f"目标: {target.shape}")
print(f"损失值: {loss.item():.4f}")

print("\n【5.3 二元交叉熵 (BCELoss)】")
print("用于二分类问题")
pred = torch.sigmoid(torch.randn(4))
target = torch.tensor([1.0, 0.0, 1.0, 0.0])
criterion = nn.BCELoss()
loss = criterion(pred, target)
print(f"预测: {pred}")
print(f"目标: {target}")
print(f"损失值: {loss.item():.4f}")


# ============================================================
# 6. 优化器
# ============================================================
print_section("6. 优化器")

# 创建一个简单的模型
model = nn.Linear(10, 1)

print("【6.1 SGD优化器】")
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(f"SGD: lr=0.01, momentum=0.9")

print("\n【6.2 Adam优化器】")
optimizer_adam = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
print(f"Adam: lr=0.001")

print("\n【6.3 优化器使用步骤】")
print("1. 清零梯度: optimizer.zero_grad()")
print("2. 反向传播: loss.backward()")
print("3. 更新参数: optimizer.step()")

# 示例
x = torch.randn(4, 10)
target = torch.randn(4, 1)

# 前向传播
output = model(x)
loss = F.mse_loss(output, target)

# 反向传播和优化
optimizer_adam.zero_grad()
loss.backward()
optimizer_adam.step()

print(f"\n损失值: {loss.item():.4f}")


# ============================================================
# 7. 完整的CNN示例
# ============================================================
print_section("7. 完整的CNN示例 - LeNet风格")


class CNN(nn.Module):
    """简单的卷积神经网络"""
    
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # 卷积层
        self.features = nn.Sequential(
            # 第一层卷积: 1 -> 6通道, 5x5卷积核
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
            
            # 第二层卷积: 6 -> 16通道
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 5x5
        )
        
        # 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 创建模型
model = CNN(num_classes=10)
print("CNN模型结构:")
print(model)

# 测试
x = torch.randn(4, 1, 28, 28)  # MNIST格式的输入
output = model(x)
print(f"\n输入: {x.shape}")
print(f"输出: {output.shape}")

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")


# ============================================================
# 8. 模型保存和加载
# ============================================================
print_section("8. 模型保存和加载")

# 创建一个简单模型
model = SimpleNet(784, 256, 10)

print("【8.1 保存整个模型】")
# torch.save(model, 'model_complete.pth')
print("torch.save(model, 'model_complete.pth')")

print("\n【8.2 只保存模型参数(推荐)】")
# torch.save(model.state_dict(), 'model_params.pth')
print("torch.save(model.state_dict(), 'model_params.pth')")

print("\n【8.3 加载模型参数】")
# model = SimpleNet(784, 256, 10)
# model.load_state_dict(torch.load('model_params.pth'))
print("model.load_state_dict(torch.load('model_params.pth'))")

print("\n【8.4 保存和加载检查点(包含优化器状态)】")
checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.SGD(model.parameters(), lr=0.01).state_dict(),
    'loss': 0.5,
}
# torch.save(checkpoint, 'checkpoint.pth')
print("保存检查点: 包含epoch、模型参数、优化器状态、损失等")


# ============================================================
# 9. 设备管理(GPU/CPU)
# ============================================================
print_section("9. 设备管理(GPU/CPU)")

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模型并移动到设备
model = SimpleNet(784, 256, 10).to(device)
print(f"模型设备: {next(model.parameters()).device}")

# 创建数据并移动到设备
x = torch.randn(4, 784).to(device)
print(f"数据设备: {x.device}")

# 前向传播
output = model(x)
print(f"输出设备: {output.device}")

# 移回CPU
output_cpu = output.cpu()
print(f"移回CPU后: {output_cpu.device}")


# ============================================================
# 10. 实用技巧
# ============================================================
print_section("10. 实用技巧")

print("【10.1 模型评估模式】")
model.train()   # 训练模式(启用Dropout和BatchNorm统计)
model.eval()    # 评估模式(关闭Dropout,使用运行统计)

print("\n【10.2 冻结参数】")
for param in model.parameters():
    param.requires_grad = False
print("所有参数已冻结(不参与训练)")

print("\n【10.3 打印模型摘要】")
from torchsummary import summary
# summary(model, input_size=(784,), device='cpu')
print("使用 torchsummary.summary(model, input_size=(784,))")

print("\n【10.4 学习率调整】")
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
print("每10个epoch学习率乘以0.1")


print("\n" + "=" * 60)
print("  神经网络构建教程完成!")
print("=" * 60)
print("\n关键要点:")
print("1. 继承nn.Module创建网络")
print("2. 在__init__中定义层,在forward中定义前向传播")
print("3. 使用nn.Sequential快速构建简单网络")
print("4. 选择合适的损失函数和优化器")
print("5. 使用.to(device)管理GPU/CPU")
print("6. 保存和加载模型参数进行持久化")
