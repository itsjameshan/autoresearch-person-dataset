@echo off
REM 自动迭代实验流程 - Windows GPU
REM 使用方式: run_all_experiments.bat

echo ========================================
echo AutoResearch - 自动迭代优化实验
echo ========================================
echo.

REM 检查环境
call conda activate ultralytics_object
if errorlevel 1 (
    echo 错误: 无法激活conda环境
    pause
    exit /b 1
)

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
if errorlevel 1 (
    echo 警告: PyTorch 或 CUDA 可能有问题
)

echo.
echo 开始实验序列...
echo ========================================
echo.

REM 实验1: yolo12m 基线
echo [1/3] 运行实验1: yolo12m 基线...
python train_with_config.py exp01_config.py
if errorlevel 1 (
    echo 实验1 失败!
    pause
    exit /b 1
)

echo.
echo ========================================
echo.

REM 实验2: Copy-Paste增强
echo [2/3] 运行实验2: Copy-Paste增强...
python train_with_config.py exp02_config.py
if errorlevel 1 (
    echo 实验2 失败!
    pause
    exit /b 1
)

echo.
echo ========================================
echo.

REM 实验3: 小目标优化
echo [3/3] 运行实验3: 小目标优化...
python train_with_config.py exp03_config.py
if errorlevel 1 (
    echo 实验3 失败!
    pause
    exit /b 1
)

echo.
echo ========================================
echo 所有实验完成!
echo ========================================
echo.
echo 请查看 autoresearch_runs/ 目录获取结果
echo.

pause