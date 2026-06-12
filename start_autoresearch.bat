@echo off
REM ========================================
REM AutoResearch Loop - Windows 启动脚本
REM ========================================

echo.
echo ========================================
echo    KARPATHY-STYLE AUTORESEARCH LOOP
echo ========================================
echo.

REM 检查 Ollama 是否正在运行
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

REM 检查 Ollama 模型
echo.
echo [2/5] 检查 Ollama 模型...
ollama list 2>NUL | find "gemma3:4b" >NUL
if "%ERRORLEVEL%"=="1" (
    echo.
    echo ⚠️ 未找到 gemma3:4b 模型
    echo.
    echo 正在拉取模型（可能需要几分钟）...
    ollama pull gemma3:4b
    if "%ERRORLEVEL%"=="1" (
        echo.
        echo ❌ 模型拉取失败
        pause
        exit /B 1
    )
)

echo ✅ 模型已准备就绪

REM 检查依赖
echo.
echo [3/5] 检查 Python 依赖...
python -c "import httpx; import psutil" 2>NUL
if "%ERRORLEVEL%"=="1" (
    echo.
    echo ⚠️ 安装缺失依赖...
    pip install httpx psutil
)

echo ✅ 依赖已就绪

REM 切换到正确目录
echo.
echo [4/5] 切换到项目目录...
cd /D "%~dp0"
if not exist "train.py" (
    echo ❌ 找不到 train.py，当前目录不对！
    pause
    exit /B 1
)

echo ✅ 目录正确

REM 启动主程序
echo.
echo [5/5] 启动自主研究循环...
echo.
echo ========================================
echo    研究循环已启动！
echo ========================================
echo.
echo 监控命令 (本终端显示训练实时输出 + agent 决策):
echo   - 状态概览: type status.md
echo   - 完整历史: type results.tsv
echo   - 训练日志: type run.log              ^(纯训练 stdout^)
echo   - 活动事件: type activity_events.jsonl ^(orchestrator 决策事件流^)
echo.
echo   另开终端实时跟踪:
echo     PowerShell:  Get-Content run.log -Wait -Tail 30
echo     PowerShell:  Get-Content activity_events.jsonl -Wait -Tail 30
echo.
echo 按 Ctrl+C 停止研究循环
echo.
echo ========================================
echo.

python autoresearch_loop.py

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
