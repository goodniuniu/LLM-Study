"""
PyTorch基础教程 - 数据加载和处理
本教程介绍如何使用Dataset和DataLoader加载数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np


def print_section(title):
    """打印章节标题"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


# ============================================================
# 1. 自定义Dataset
# ============================================================
print_section("1. 自定义Dataset")


class SimpleDataset(Dataset):
    """
    简单的自定义Dataset示例
    用于理解Dataset的工作原理
    """
    
    def __init__(self, data, labels, transform=None):
        """
        初始化数据集
        Args:
            data: 数据,形状为 (N, ...)
            labels: 标签,形状为 (N,)
            transform: 数据变换
        """
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        Args:
            idx: 样本索引
        Returns:
            sample: 样本数据
            label: 样本标签
        """
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label


# 创建示例数据
np.random.seed(42)
data = np.random.randn(100, 10).astype(np.float32)  # 100个样本,每个10维
labels = np.random.randint(0, 3, 100)  # 3分类

# 创建Dataset
dataset = SimpleDataset(data, labels)
print(f"数据集大小: {len(dataset)}")

# 获取单个样本
sample, label = dataset[0]
print(f"第一个样本: 形状={sample.shape}, 标签={label}")

# 遍历数据集
print("\n前5个样本:")
for i in range(5):
    sample, label = dataset[i]
    print(f"  样本{i}: 形状={sample.shape}, 标签={label}")


# ============================================================
# 2. 使用DataLoader
# ============================================================
print_section("2. 使用DataLoader")

# 创建DataLoader
dataloader = DataLoader(
    dataset=dataset,
    batch_size=8,      # 每批8个样本
    shuffle=True,      # 打乱数据
    num_workers=0,     # 数据加载线程数(Windows建议设为0)
    drop_last=False    # 是否丢弃最后不完整的批次
)

print(f"DataLoader配置:")
print(f"  批次大小: 8")
print(f"  打乱数据: True")
print(f"  总批次数: {len(dataloader)}")

# 遍历DataLoader
print("\n前3个批次:")
for batch_idx, (data_batch, label_batch) in enumerate(dataloader):
    print(f"  批次{batch_idx}: 数据形状={data_batch.shape}, 标签形状={label_batch.shape}")
    if batch_idx >= 2:
        break


# ============================================================
# 3. 图像数据变换 (Transforms)
# ============================================================
print_section("3. 图像数据变换 (Transforms)")

print("【3.1 基础变换】")

# 组合多个变换
transform = transforms.Compose([
    transforms.ToTensor(),  # PIL Image -> Tensor, 并归一化到[0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1,1]
])

print("Compose变换:")
print("  1. ToTensor(): 转换为张量")
print("  2. Normalize(): 标准化")

print("\n【3.2 数据增强变换】")

augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=15),      # 随机旋转
    transforms.RandomHorizontalFlip(p=0.5),     # 随机水平翻转
    transforms.RandomCrop(size=28, padding=4),  # 随机裁剪
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("数据增强:")
print("  - RandomRotation: 随机旋转")
print("  - RandomHorizontalFlip: 随机水平翻转")
print("  - RandomCrop: 随机裁剪")
print("  - ColorJitter: 颜色抖动")

print("\n【3.3 自定义变换】")


class AddNoise(object):
    """自定义变换: 添加噪声"""
    
    def __init__(self, noise_factor=0.1):
        self.noise_factor = noise_factor
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_factor
        return tensor + noise


# 使用自定义变换
custom_transform = transforms.Compose([
    transforms.ToTensor(),
    AddNoise(noise_factor=0.1),
    transforms.Normalize((0.5,), (0.5,))
])

print("自定义变换AddNoise: 添加高斯噪声")


# ============================================================
# 4. 完整的图像Dataset示例
# ============================================================
print_section("4. 完整的图像Dataset示例")


class ImageDataset(Dataset):
    """图像数据集示例"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label


print("图像Dataset结构:")
print("  - 接收图像路径列表和标签")
print("  - __getitem__中加载图像并应用变换")
print("  - 返回处理后的图像和标签")


# ============================================================
# 5. 使用内置数据集 - MNIST
# ============================================================
print_section("5. 使用内置数据集 - MNIST")

# 定义变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST的标准化参数
])

# 下载并加载MNIST数据集
print("下载MNIST数据集...")
train_dataset = datasets.MNIST(
    root='../../data/raw',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='../../data/raw',
    train=False,
    download=True,
    transform=transform
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"类别数: {len(train_dataset.classes)}")
print(f"类别名称: {train_dataset.classes}")

# 查看单个样本
image, label = train_dataset[0]
print(f"\n第一个样本:")
print(f"  图像形状: {image.shape}")
print(f"  标签: {label}")
print(f"  像素值范围: [{image.min():.2f}, {image.max():.2f}]")

# 创建DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

print(f"\n训练DataLoader批次数: {len(train_loader)}")
print(f"测试DataLoader批次数: {len(test_loader)}")

# 获取一个批次
images, labels = next(iter(train_loader))
print(f"\n一个批次:")
print(f"  图像批次形状: {images.shape}")  # (64, 1, 28, 28)
print(f"  标签批次形状: {labels.shape}")  # (64,)


# ============================================================
# 6. 高级DataLoader技巧
# ============================================================
print_section("6. 高级DataLoader技巧")

print("【6.1 自定义 collate_fn】")


def custom_collate(batch):
    """
    自定义批次处理函数
    用于处理变长数据等特殊场景
    """
    data = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # 将数据堆叠成批次
    data = torch.stack(data, dim=0)
    labels = torch.tensor(labels)
    
    return data, labels


# 使用自定义collate_fn
dataloader_custom = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=custom_collate
)

print("自定义collate_fn可以处理:")
print("  - 变长序列(需要填充)")
print("  - 特殊的数据格式")
print("  - 数据预处理")

print("\n【6.2 Sampler - 控制采样方式】")

from torch.utils.data import SubsetRandomSampler

# 创建索引
indices = list(range(len(dataset)))
np.random.shuffle(indices)

# 划分训练集和验证集
split = int(0.8 * len(dataset))
train_idx, val_idx = indices[:split], indices[split:]

# 创建Sampler
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)

# 使用Sampler创建DataLoader
train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=8, sampler=val_sampler)

print(f"使用Sampler划分数据集:")
print(f"  训练集大小: {len(train_idx)}")
print(f"  验证集大小: {len(val_idx)}")

print("\n【6.3 WeightedRandomSampler - 处理类别不平衡】")

from torch.utils.data import WeightedRandomSampler

# 假设有类别不平衡问题
class_counts = [10, 50, 100]  # 3个类别的样本数
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)

# 为每个样本计算权重
sample_weights = []
for label in labels:
    sample_weights.append(class_weights[label])

# 创建WeightedRandomSampler
weighted_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

print("WeightedRandomSampler:")
print("  - 给少数类更高的采样权重")
print("  - 解决类别不平衡问题")


# ============================================================
# 7. 数据加载最佳实践
# ============================================================
print_section("7. 数据加载最佳实践")

print("【7.1 数据预处理建议】")
print("  1. 在__getitem__中进行实时预处理")
print("  2. 使用transforms.Compose组合多个变换")
print("  3. 数据增强只在训练时应用")
print("  4. 归一化参数使用预训练模型的统计值")

print("\n【7.2 DataLoader优化建议】")
print("  1. 根据GPU内存调整batch_size")
print("  2. 设置num_workers>0加速数据加载(Linux/Mac)")
print("  3. 使用pin_memory=True加速GPU数据传输")
print("  4. 使用persistent_workers减少进程创建开销")

print("\n【7.3 内存优化建议】")
print("  1. 使用生成器延迟加载数据")
print("  2. 大图先resize再加载")
print("  3. 使用LMDB等数据库存储数据")
print("  4. 及时释放不需要的数据")


# ============================================================
# 8. 完整的训练数据管道示例
# ============================================================
print_section("8. 完整的训练数据管道示例")


def create_data_loaders(data_dir, batch_size=32):
    """
    创建训练和验证数据加载器
    """
    # 训练数据变换(包含数据增强)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 验证数据变换(不包含数据增强)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 创建数据集(这里使用ImageFolder,假设数据按文件夹组织)
    # train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transform)
    # val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=val_transform)
    
    # 创建DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print("数据管道创建完成:")
    print(f"  训练变换: 包含数据增强")
    print(f"  验证变换: 不包含数据增强")
    print(f"  批次大小: {batch_size}")
    
    # return train_loader, val_loader


create_data_loaders("./data", batch_size=32)


# ============================================================
# 9. 处理特殊数据类型
# ============================================================
print_section("9. 处理特殊数据类型")

print("【9.1 文本数据】")
print("""
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer

# 加载文本数据
train_iter = IMDB(split='train')
tokenizer = get_tokenizer('basic_english')

# 构建词汇表
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
""")

print("\n【9.2 序列数据(如时间序列)】")


class SequenceDataset(Dataset):
    """序列数据Dataset示例"""
    
    def __init__(self, sequences, labels, seq_length):
        self.sequences = sequences
        self.labels = labels
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # 截断或填充到固定长度
        if len(seq) > self.seq_length:
            seq = seq[:self.seq_length]
        else:
            seq = seq + [0] * (self.seq_length - len(seq))
        
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label)


print("序列Dataset:")
print("  - 处理变长序列")
print("  - 需要截断或填充")


print("\n" + "=" * 60)
print("  数据加载教程完成!")
print("=" * 60)
print("\n关键要点:")
print("1. 继承Dataset类自定义数据集")
print("2. 使用DataLoader批量加载数据")
print("3. 使用transforms进行数据预处理")
print("4. 训练集和验证集使用不同的变换")
print("5. 使用Sampler控制采样策略")
print("6. 合理设置num_workers和batch_size")
