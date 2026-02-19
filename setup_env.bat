@echo off
echo ========================================
echo 深度学习学习项目 - 虚拟环境设置脚本
echo ========================================
echo.

REM 检查虚拟环境是否存在
if not exist "venv" (
    echo [1/5] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo 错误: 无法创建虚拟环境
        pause
        exit /b 1
    )
    echo 虚拟环境创建成功!
) else (
    echo 虚拟环境已存在
)

echo.
echo [2/5] 激活虚拟环境并升级pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo [3/5] 安装基础依赖包...
pip install numpy pandas scikit-learn tqdm ipython matplotlib

echo.
echo [4/5] 安装PyTorch (CPU版本)...
REM 注意: 如果需要GPU版本,请访问 https://pytorch.org/ 获取安装命令
pip install torch torchvision torchaudio

echo.
echo [5/5] 安装NLP和其他依赖...
pip install transformers tokenizers datasets accelerate peft seaborn tensorboard jupyter black

echo.
echo ========================================
echo 安装完成!
echo ========================================
echo.
echo 使用方法:
echo   1. 激活虚拟环境: venv\Scripts\activate.bat
echo   2. 退出虚拟环境: deactivate
echo.
echo 如需安装GPU版本PyTorch,请访问:
echo   https://pytorch.org/get-started/locally/
echo.
pause
