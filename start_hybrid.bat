@echo off
REM ========================================
REM Hybrid AutoResearch - Windows 启动脚本
REM ========================================

echo.
echo ========================================
echo    HYBRID AUTORESEARCH LOOP
echo ========================================
echo.

REM 检查 Ollama
echo [1/5] 检查 Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo.
    echo ⚠️ Ollama 未运行！
    echo.
    echo 请在另一个终端运行:
    echo   $env:OLLAMA_GPU_LAYERS = 0; ollama serve
    echo.
    pause
    exit /B 1
)
echo ✅ Ollama 正在运行

REM 检查模型
echo.
echo [2/5] 检查 Ollama 模型...
ollama list 2>NUL | find "gemma3:4b" >NUL
if "%ERRORLEVEL%"=="1" (
    echo.
    echo ⚠️ 未找到 gemma3:4b 模型
    echo.
    echo 正在拉取模型...
    ollama pull gemma3:4b
    if "%ERRORLEVEL%"=="1" (
        echo ❌ 模型拉取失败
        pause
        exit /B 1
    )
)
echo ✅ 模型已就绪

REM 检查依赖
echo.
echo [3/5] 检查 Python 依赖...
python -c "import httpx; import psutil; import skopt" 2>NUL
if "%ERRORLEVEL%"=="1" (
    echo.
    echo ⚠️ 安装缺失依赖...
    pip install httpx psutil scikit-optimize
)
echo ✅ 依赖已就绪

REM 切换目录
echo.
echo [4/5] 切换到项目目录...
cd /D "%~dp0"
if not exist "train.py" (
    echo ❌ 找不到 train.py！
    pause
    exit /B 1
)
echo ✅ 目录正确

REM 启动
echo.
echo [5/5] 启动混合自主研究循环...
echo.
echo ========================================
echo    HYBRID LOOP 已启动！
echo ========================================
echo.
echo 阶段说明:
echo   - Phase 1 (📊 BAYESIAN): 前 12 次实验，贝叶斯快速搜索
echo   - Phase 2 (🧠 LLM): 之后实验，LLM 深度研究
echo.
echo 监控命令:
echo   - 状态: type status.md
echo   - 历史: type results.tsv
echo   - 日志: type run.log
echo.
echo 按 Ctrl+C 停止
echo.
echo ========================================
echo.

python hybrid_loop.py

if "%ERRORLEVEL%"=="0" (
    echo.
    echo ========================================
    echo    研究完成！
    echo ========================================
) else (
    echo.
    echo ========================================
    echo    研究过程出错（退出码 %ERRORLEVEL%）
    echo ========================================
)

echo.
pause
