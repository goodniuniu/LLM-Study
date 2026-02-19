#!/usr/bin/env python3
"""
环境检查脚本 - 验证深度学习环境是否完备
"""

import sys


def check_basic():
    """检查基础依赖"""
    print("\n[基础依赖]")
    try:
        import numpy
        print(f"  [OK] NumPy: {numpy.__version__}")
    except ImportError:
        print("  [X] NumPy: 未安装")
    
    try:
        import pandas
        print(f"  [OK] Pandas: {pandas.__version__}")
    except ImportError:
        print("  [X] Pandas: 未安装")
    
    try:
        import sklearn
        print(f"  [OK] Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("  [X] Scikit-learn: 未安装")
    
    try:
        import tqdm
        print(f"  [OK] tqdm: {tqdm.__version__}")
    except ImportError:
        print("  [X] tqdm: 未安装")


def check_pytorch():
    """检查PyTorch"""
    print("\n[PyTorch]")
    try:
        import torch
        print(f"  [OK] PyTorch: {torch.__version__}")
        print(f"  [INFO] CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  [INFO] CUDA版本: {torch.version.cuda}")
            print(f"  [INFO] GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  [INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("  [X] PyTorch: 未安装")
    
    try:
        import torchvision
        print(f"  [OK] TorchVision: {torchvision.__version__}")
    except ImportError:
        print("  [X] TorchVision: 未安装")
    
    try:
        import torchaudio
        print(f"  [OK] TorchAudio: {torchaudio.__version__}")
    except ImportError:
        print("  [X] TorchAudio: 未安装")


def check_nlp():
    """检查NLP依赖"""
    print("\n[NLP库]")
    try:
        import transformers
        print(f"  [OK] Transformers: {transformers.__version__}")
    except ImportError:
        print("  [X] Transformers: 未安装")
    
    try:
        import datasets
        print(f"  [OK] Datasets: {datasets.__version__}")
    except ImportError:
        print("  [X] Datasets: 未安装")
    
    try:
        import tokenizers
        print(f"  [OK] Tokenizers: {tokenizers.__version__}")
    except ImportError:
        print("  [X] Tokenizers: 未安装")
    
    try:
        import accelerate
        print(f"  [OK] Accelerate: {accelerate.__version__}")
    except ImportError:
        print("  [X] Accelerate: 未安装")
    
    try:
        import peft
        print(f"  [OK] PEFT: {peft.__version__}")
    except ImportError:
        print("  [X] PEFT: 未安装")


def check_visualization():
    """检查可视化库"""
    print("\n[可视化]")
    try:
        import matplotlib
        print(f"  [OK] Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("  [X] Matplotlib: 未安装")
    
    try:
        import seaborn
        print(f"  [OK] Seaborn: {seaborn.__version__}")
    except ImportError:
        print("  [X] Seaborn: 未安装")
    
    try:
        import tensorboard
        print(f"  [OK] TensorBoard: 已安装")
    except ImportError:
        print("  [X] TensorBoard: 未安装")


def check_dev_tools():
    """检查开发工具"""
    print("\n[开发工具]")
    try:
        import jupyter
        print(f"  [OK] Jupyter: 已安装")
    except ImportError:
        print("  [X] Jupyter: 未安装")
    
    try:
        import ipython
        print(f"  [OK] IPython: {ipython.__version__}")
    except ImportError:
        print("  [X] IPython: 未安装")
    
    try:
        import black
        print(f"  [OK] Black: {black.__version__}")
    except ImportError:
        print("  [X] Black: 未安装")


def test_pytorch():
    """测试PyTorch基本功能"""
    print("\n[PyTorch功能测试]")
    try:
        import torch
        # 创建张量
        x = torch.rand(5, 3)
        print(f"  [OK] 张量创建: {x.shape}")
        
        # 矩阵乘法
        y = torch.rand(3, 4)
        z = torch.mm(x, y)
        print(f"  [OK] 矩阵乘法: {z.shape}")
        
        # 自动求导
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2
        y.backward()
        print(f"  [OK] 自动求导: dy/dx = {x.grad.item()}")
        
        print("  [OK] PyTorch功能正常!")
    except Exception as e:
        print(f"  [X] PyTorch测试失败: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("深度学习环境检查报告")
    print("=" * 60)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    
    check_basic()
    check_pytorch()
    check_nlp()
    check_visualization()
    check_dev_tools()
    test_pytorch()
    
    print("\n" + "=" * 60)
    print("环境检查完成!")
    print("=" * 60)
    print("\n提示: 可以开始学习深度学习了!")
    print("   建议从 01-basics/python-basics 开始")


if __name__ == "__main__":
    main()
