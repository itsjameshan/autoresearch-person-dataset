@echo off
REM Windows GPU Training Launcher for Autoresearch
REM Run this on a Windows machine with CUDA GPU

echo ========================================
echo Autoresearch - YOLO Training Launcher
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Conda not found. Please install Miniconda or Anaconda.
    pause
    exit /b 1
)

REM Activate ultralytics environment
call conda activate ultralytics_object

REM Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo.
echo Starting training...
echo.

REM Run training with logging
python train_gpu.py > run.log 2>&1

echo.
echo Training completed. Check run.log for details.
echo.

REM Display last few lines of results
if exist last_metrics.json (
    echo ========================================
    echo Last Metrics:
    type last_metrics.json
    echo ========================================
)

pause