# 快速开始指南

## 1. 创建虚拟环境

### 方式一: 使用自动脚本(推荐)

**Windows用户:**
```batch
# 双击运行或命令行执行
setup_env.bat
```

**PowerShell用户:**
```powershell
.\setup_env.ps1
```

### 方式二: 手动创建

```batch
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
venv\Scripts\activate

# 3. 升级pip
python -m pip install --upgrade pip

# 4. 安装基础依赖
pip install numpy pandas scikit-learn tqdm ipython matplotlib

# 5. 安装PyTorch (CPU版本)
pip install torch torchvision torchaudio

# 6. 安装NLP和其他依赖
pip install transformers tokenizers datasets accelerate peft seaborn tensorboard jupyter black
```

## 2. 激活虚拟环境

```batch
# Windows命令行
venv\Scripts\activate

# 或使用快捷脚本
activate.bat

# PowerShell
.\venv\Scripts\Activate.ps1
```

## 3. 退出虚拟环境

```batch
deactivate
```

## 4. 验证安装

```python
import torch
import numpy as np
import transformers

print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())
print("NumPy版本:", np.__version__)
print("Transformers版本:", transformers.__version__)
```

## 5. 开始学习

从 `01-basics/python-basics` 开始你的学习之旅!

## GPU版本PyTorch安装

如需安装GPU版本PyTorch,请访问:
https://pytorch.org/get-started/locally/

根据你的CUDA版本选择对应的安装命令。

## 常见问题

### Q: pip下载速度慢怎么办?
A: 使用国内镜像源:
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 包名
```

### Q: 如何查看已安装的包?
```bash
pip list
```

### Q: 如何导出/导入依赖?
```bash
# 导出
pip freeze > requirements-export.txt

# 导入
pip install -r requirements-export.txt
```
