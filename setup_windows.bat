@echo off
REM ============================================================
REM Autoresearch Windows Setup Script
REM RTX 5070 12GB / 64GB RAM / i5-14600KF
REM ============================================================

echo === Autoresearch Windows Setup ===
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Check CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if errorlevel 1 (
    echo Installing PyTorch with CUDA 12.4 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
)

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Check Ollama
ollama --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Ollama not found.
    echo Download from: https://ollama.com/download
    echo After installing, run:
    echo   ollama serve
    echo   ollama pull qwen2.5-coder:14b
    echo.
) else (
    echo Pulling Ollama model...
    ollama pull qwen2.5-coder:14b
)

REM Check dataset
if exist "D:\PythonProject\person_dataset\images\train" (
    echo Dataset found at D:\PythonProject\person_dataset
) else (
    echo.
    echo WARNING: Dataset not found at D:\PythonProject\person_dataset
    echo Make sure images/ and labels/ directories exist there.
    echo.
)

REM Check git branch
git branch --show-current
echo.

echo === Setup Complete ===
echo.
echo To start the autonomous experiment loop:
echo   1. Start Ollama in another terminal: ollama serve
echo   2. Run: python ollama_runner.py
echo.
pause
