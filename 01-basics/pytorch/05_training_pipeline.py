"""
PyTorch基础教程 - 完整训练流程
本教程展示一个完整的深度学习训练流程,包括训练、验证、测试和保存模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


# ============================================================
# 1. 准备数据
# ============================================================
print_section("1. 准备数据")

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 生成合成数据(模拟二分类问题)
# 类别0: 中心在(2, 2)
# 类别1: 中心在(-2, -2)
n_samples = 1000

# 类别0
X_0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
y_0 = np.zeros(n_samples // 2)

# 类别1
X_1 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
y_1 = np.ones(n_samples // 2)

# 合并数据
X = np.vstack([X_0, X_1]).astype(np.float32)
y = np.hstack([y_0, y_1]).astype(np.int64)

# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")
print(f"特征维度: {X_train.shape[1]}")
print(f"类别数: {len(np.unique(y))}")

# 创建DataLoader
train_dataset = TensorDataset(
    torch.from_numpy(X_train),
    torch.from_numpy(y_train)
)

test_dataset = TensorDataset(
    torch.from_numpy(X_test),
    torch.from_numpy(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ============================================================
# 2. 定义模型
# ============================================================
print_section("2. 定义模型")


class SimpleClassifier(nn.Module):
    """简单的二分类神经网络"""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x


# 创建模型
input_size = 2
hidden_size = 64
num_classes = 2

model = SimpleClassifier(input_size, hidden_size, num_classes)
print("模型结构:")
print(model)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,}")


# ============================================================
# 3. 定义损失函数和优化器
# ============================================================
print_section("3. 定义损失函数和优化器")

# 损失函数
criterion = nn.CrossEntropyLoss()
print(f"损失函数: {criterion}")

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
print(f"优化器: {optimizer}")
print(f"  学习率: 0.001")
print(f"  权重衰减: 1e-5")

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
print(f"学习率调度器: StepLR")
print(f"  每30个epoch学习率乘以0.1")


# ============================================================
# 4. 训练循环
# ============================================================
print_section("4. 训练循环")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移到设备
        data, target = data.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """
    评估模型
    
    Args:
        model: 神经网络模型
        data_loader: 数据加载器
        criterion: 损失函数
        device: 计算设备
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()  # 设置为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 统计
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}\n")

# 将模型移到设备
model = model.to(device)

# 训练参数
num_epochs = 50
best_accuracy = 0.0

# 训练历史
history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
}

print("开始训练...")
print("-" * 60)

for epoch in range(num_epochs):
    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # 测试
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    # 更新学习率
    scheduler.step()
    
    # 记录历史
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    
    # 保存最佳模型
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    # 打印进度
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"  测试 - 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print()

print("-" * 60)
print(f"训练完成! 最佳测试准确率: {best_accuracy:.2f}%")


# ============================================================
# 5. 模型评估
# ============================================================
print_section("5. 模型评估")

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 在测试集上评估
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"最佳模型在测试集上的表现:")
print(f"  损失: {test_loss:.4f}")
print(f"  准确率: {test_acc:.2f}%")

# 详细分类报告
from sklearn.metrics import classification_report, confusion_matrix

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        _, predicted = output.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(target.numpy())

print("\n分类报告:")
print(classification_report(all_targets, all_preds, target_names=['Class 0', 'Class 1']))

print("混淆矩阵:")
print(confusion_matrix(all_targets, all_preds))


# ============================================================
# 6. 模型保存和加载
# ============================================================
print_section("6. 模型保存和加载")

print("【6.1 保存模型参数】")
torch.save(model.state_dict(), 'model_params.pth')
print("模型参数已保存到: model_params.pth")

print("\n【6.2 保存完整模型】")
torch.save(model, 'model_complete.pth')
print("完整模型已保存到: model_complete.pth")

print("\n【6.3 保存检查点】")
checkpoint = {
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_accuracy': best_accuracy,
    'history': history
}
torch.save(checkpoint, 'checkpoint.pth')
print("检查点已保存到: checkpoint.pth")

print("\n【6.4 加载模型参数】")
# 创建新模型实例
new_model = SimpleClassifier(input_size, hidden_size, num_classes)
new_model.load_state_dict(torch.load('model_params.pth'))
print("模型参数已加载")

print("\n【6.5 从检查点恢复训练】")
checkpoint = torch.load('checkpoint.pth')
new_model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
best_acc = checkpoint['best_accuracy']
print(f"从epoch {start_epoch}恢复,最佳准确率: {best_acc:.2f}%")


# ============================================================
# 7. 可视化训练过程
# ============================================================
print_section("7. 可视化训练过程")

print("训练历史:")
print(f"  最终训练准确率: {history['train_acc'][-1]:.2f}%")
print(f"  最终测试准确率: {history['test_acc'][-1]:.2f}%")
print(f"  训练损失下降: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")
print(f"  测试损失下降: {history['test_loss'][0]:.4f} -> {history['test_loss'][-1]:.4f}")

# 绘制训练曲线(如果matplotlib可用)
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['test_loss'], label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(history['train_acc'], label='Train Accuracy')
    axes[1].plot(history['test_acc'], label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("\n训练曲线已保存到: training_history.png")
    
except ImportError:
    print("\nmatplotlib未安装,跳过可视化")


# ============================================================
# 8. 模型推理
# ============================================================
print_section("8. 模型推理")


def predict(model, input_data, device):
    """
    对输入数据进行预测
    
    Args:
        model: 训练好的模型
        input_data: 输入数据 (numpy array 或 tensor)
        device: 计算设备
    
    Returns:
        prediction: 预测类别
        probability: 预测概率
    """
    model.eval()
    
    # 转换为tensor
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.from_numpy(input_data).float()
    
    # 添加批次维度(如果需要)
    if input_data.dim() == 1:
        input_data = input_data.unsqueeze(0)
    
    input_data = input_data.to(device)
    
    with torch.no_grad():
        output = model(input_data)
        probability = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1)
    
    return prediction.cpu().numpy(), probability.cpu().numpy()


# 测试推理
test_samples = torch.tensor([
    [2.0, 2.0],   # 应该预测为类别0
    [-2.0, -2.0], # 应该预测为类别1
    [1.5, 1.8],   # 应该预测为类别0
    [-1.5, -1.8]  # 应该预测为类别1
], dtype=torch.float32)

print("测试样本:")
for i, sample in enumerate(test_samples):
    pred, prob = predict(model, sample, device)
    print(f"  样本 {i+1}: {sample.numpy()}")
    print(f"    预测类别: {pred[0]}, 概率: {prob[0]}")


# ============================================================
# 9. 训练技巧总结
# ============================================================
print_section("9. 训练技巧总结")

print("【9.1 防止过拟合】")
print("  - 使用Dropout层")
print("  - 添加权重衰减(L2正则化)")
print("  - 早停(Early Stopping)")
print("  - 数据增强")

print("\n【9.2 加速训练】")
print("  - 使用GPU加速")
print("  - 调整batch_size")
print("  - 使用混合精度训练")
print("  - 使用更高效的数据加载")

print("\n【9.3 提高准确率】")
print("  - 调整学习率和学习率调度")
print("  - 尝试不同的优化器")
print("  - 调整网络架构")
print("  - 使用预训练模型")

print("\n【9.4 调试技巧】")
print("  - 从小数据集开始测试")
print("  - 检查损失是否正常下降")
print("  - 监控梯度和权重")
print("  - 可视化中间层输出")


# 清理临时文件
print("\n清理临时文件...")
for file in ['best_model.pth', 'model_params.pth', 'model_complete.pth', 'checkpoint.pth']:
    if os.path.exists(file):
        os.remove(file)
        print(f"  已删除: {file}")


print("\n" + "=" * 60)
print("  完整训练流程教程完成!")
print("=" * 60)
print("\n关键要点:")
print("1. 使用train()和eval()切换模式")
print("2. 训练时zero_grad() -> backward() -> step()")
print("3. 验证时使用torch.no_grad()节省内存")
print("4. 保存最佳模型避免过拟合")
print("5. 使用学习率调度器调整学习率")
print("6. 记录训练历史用于分析")
