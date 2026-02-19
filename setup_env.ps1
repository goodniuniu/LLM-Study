# PowerShell环境设置脚本
Write-Host "========================================" -ForegroundColor Green
Write-Host "深度学习学习项目 - 虚拟环境设置脚本" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 检查虚拟环境是否存在
if (-not (Test-Path "venv")) {
    Write-Host "[1/5] 创建虚拟环境..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误: 无法创建虚拟环境" -ForegroundColor Red
        Read-Host "按Enter键退出"
        exit 1
    }
    Write-Host "虚拟环境创建成功!" -ForegroundColor Green
} else {
    Write-Host "虚拟环境已存在" -ForegroundColor Green
}

Write-Host ""
Write-Host "[2/5] 激活虚拟环境并升级pip..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

Write-Host ""
Write-Host "[3/5] 安装基础依赖包..." -ForegroundColor Yellow
pip install numpy pandas scikit-learn tqdm ipython matplotlib

Write-Host ""
Write-Host "[4/5] 安装PyTorch (CPU版本)..." -ForegroundColor Yellow
# 注意: 如果需要GPU版本,请访问 https://pytorch.org/ 获取安装命令
pip install torch torchvision torchaudio

Write-Host ""
Write-Host "[5/5] 安装NLP和其他依赖..." -ForegroundColor Yellow
pip install transformers tokenizers datasets accelerate peft seaborn tensorboard jupyter black

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "安装完成!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "使用方法:" -ForegroundColor Cyan
Write-Host "  1. 激活虚拟环境: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. 退出虚拟环境: deactivate" -ForegroundColor White
Write-Host ""
Write-Host "如需安装GPU版本PyTorch,请访问:" -ForegroundColor Yellow
Write-Host "  https://pytorch.org/get-started/locally/" -ForegroundColor Yellow
Write-Host ""
Read-Host "按Enter键退出"
