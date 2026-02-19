# Git 使用指南

本项目的Git管理指南

## 常用Git命令

### 查看状态
```bash
# 查看当前状态
git status

# 查看提交历史
git log --oneline

# 查看文件修改
git diff
```

### 添加和提交
```bash
# 添加所有修改
git add .

# 添加特定文件
git add filename.py

# 提交更改
git commit -m "提交信息"

# 添加并提交(快捷方式)
git commit -am "提交信息"
```

### 分支管理
```bash
# 查看分支
git branch

# 创建新分支
git branch feature-name

# 切换分支
git checkout feature-name

# 创建并切换分支
git checkout -b feature-name

# 合并分支
git checkout master
git merge feature-name

# 删除分支
git branch -d feature-name
```

## 项目提交规范

### 提交信息格式
```
<type>: <subject>

<body>
```

### 类型(type)
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式(不影响功能)
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### 示例
```bash
# 添加新教程
git commit -m "feat: 添加CNN教程

- 实现卷积神经网络基础教程
- 添加图像分类示例代码
- 包含数据增强技术"

# 修复bug
git commit -m "fix: 修复GPU检测脚本中的错误

- 修复CUDA版本检测问题
- 添加错误处理"

# 更新文档
git commit -m "docs: 更新README学习路径"
```

## 学习进度提交建议

### 每次完成一个教程后提交
```bash
# 完成Tensor基础教程后
git add 01-basics/pytorch/01_tensor_basics.py
git commit -m "feat: 完成Tensor基础教程学习

- 学习Tensor创建和操作
- 理解Tensor属性和运算
- 掌握GPU Tensor操作"
```

### 完成一个阶段后提交
```bash
# 完成PyTorch基础阶段
git add 01-basics/pytorch/
git commit -m "feat: 完成PyTorch基础学习阶段

- 完成6个PyTorch基础教程
- 掌握Tensor、Autograd、神经网络构建
- 理解数据加载和训练流程
- 学会GPU加速技术"
```

## 忽略文件说明

项目已配置 `.gitignore`，以下内容不会被提交:
- 虚拟环境 (`venv/`)
- Python缓存 (`__pycache__/`, `*.pyc`)
- 数据文件 (`data/raw/*`, `data/processed/*`)
- 模型文件 (`models/checkpoints/*`, `models/pretrained/*`)
- 日志文件 (`logs/`, `*.log`)
- Jupyter检查点 (`.ipynb_checkpoints/`)

## 远程仓库(可选)

### 关联GitHub仓库
```bash
# 添加远程仓库
git remote add origin https://github.com/yourusername/LLM-Study.git

# 推送到远程
git push -u origin master

# 后续推送
git push
```

### 从远程拉取更新
```bash
git pull origin master
```

## 常见问题

### Q: 不小心提交了不该提交的文件?
```bash
# 从暂存区移除但保留文件
git rm --cached filename

# 修改.gitignore后重新提交
git add .gitignore
git commit -m "chore: 更新.gitignore"
```

### Q: 如何查看某个文件的修改历史?
```bash
git log -p filename
```

### Q: 如何撤销上次的提交?
```bash
# 保留修改
git reset --soft HEAD~1

# 丢弃修改
git reset --hard HEAD~1
```

## 学习记录建议

建议按以下节奏提交:

1. **完成每个教程后** - 提交代码和笔记
2. **完成每个阶段后** - 总结性提交
3. **每周学习结束后** - 周总结提交
4. **重要里程碑** - 打标签(tag)

```bash
# 打标签示例
git tag -a v0.1 -m "完成PyTorch基础学习"
git push origin v0.1
```

## 当前状态

```bash
# 查看当前提交历史
git log --oneline

# 查看当前分支
git branch

# 查看工作区状态
git status
```
