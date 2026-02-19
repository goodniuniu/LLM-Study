@echo off
chcp 65001 >nul
echo ========================================
echo GPU版PyTorch安装脚本
echo ========================================
echo.

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 先卸载CPU版本
echo [1/3] 卸载CPU版PyTorch...
pip uninstall torch torchvision torchaudio -y

echo.
echo [2/3] 安装GPU版PyTorch (CUDA 12.4)...
echo 正在下载安装,请耐心等待...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo [3/3] 安装完成,验证安装...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"

echo.
echo ========================================
echo 安装完成!
echo ========================================
pause
