#!/usr/bin/env python3
"""
深度学习学习项目 - 环境检查脚本
检查虚拟环境是否配置完备
"""

import sys
import importlib

def check_package(package_name, import_name=None):
    """检查包是否已安装并返回版本"""
    if import_name is None:
        import_name = package_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', '未知')
        return True, version
    except ImportError:
        return False, None

def main():
    print("=" * 60)
    print("深度学习学习项目 - 环境检查")
    print("=" * 60)
    print()
    
    # Python版本
    print(f"Python版本: {sys.version}")
    print()
    
    # 核心依赖检查
    print("【核心依赖】")
    core_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
        ('ipython', 'IPython'),
    ]
    
    for name, import_name in core_packages:
        installed, version = check_package(name, import_name)
        status = "✓" if installed else "✗"
        print(f"  {status} {name:20s} {version}")
    
    print()
    
    # 深度学习框架
    print("【深度学习框架】")
    dl_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('torchaudio', 'torchaudio'),
    ]
    
    for name, import_name in dl_packages:
        installed, version = check_package(name, import_name)
        status = "✓" if installed else "✗"
        print(f"  {status} {name:20s} {version}")
    
    # PyTorch CUDA信息
    try:
        import torch
        print(f"  → PyTorch CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  → CUDA版本: {torch.version.cuda}")
            print(f"  → GPU数量: {torch.cuda.device_count()}")
    except:
        pass
    
    print()
    
    # NLP相关
    print("【NLP工具】")
    nlp_packages = [
        ('transformers', 'transformers'),
        ('tokenizers', 'tokenizers'),
        ('datasets', 'datasets'),
        ('accelerate', 'accelerate'),
        ('peft', 'peft'),
    ]
    
    for name, import_name in nlp_packages:
        installed, version = check_package(name, import_name)
        status = "✓" if installed else "✗"
        print(f"  {status} {name:20s} {version}")
    
    print()
    
    # 开发工具
    print("【开发工具】")
    dev_packages = [
        ('jupyter', 'jupyter'),
        ('black', 'black'),
        ('tensorboard', 'tensorboard'),
    ]
    
    for name, import_name in dev_packages:
        installed, version = check_package(name, import_name)
        status = "✓" if installed else "✗"
        print(f"  {status} {name:20s} {version}")
    
    print()
    print("=" * 60)
    print("环境检查完成!")
    print("=" * 60)
    print()
    print("提示:")
    print("  • 所有 ✓ 表示已正确安装")
    print("  • 如有 ✗ 表示缺少该包,请运行: pip install 包名")
    print("  • CUDA显示False是正常现象(使用CPU版本PyTorch)")
    print("  • 如需GPU支持,请安装CUDA版本的PyTorch")
    print()

if __name__ == "__main__":
    main()
