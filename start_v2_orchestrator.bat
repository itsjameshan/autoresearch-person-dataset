@echo off
REM ========================================
REM AutoResearch v2 (multi-agent) - Windows starter
REM
REM This terminal will show:
REM   1. Live training stdout (every line, flushed in real time)
REM   2. Orchestrator activity events (iteration boundaries, agent
REM      decisions, keep/discard, HITL gates, budget checks)
REM
REM Both streams interleave in this single window. activity_events.jsonl
REM also gets a structured replay log for tools/dashboards.
REM ========================================

setlocal enabledelayedexpansion

echo.
echo ========================================
echo    AUTORESEARCH v2 - MULTI AGENT
echo ========================================
echo.

REM 1. Ollama check
echo [1/4] checking Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo.
    echo [WARN] Ollama is not running. Start it in another terminal:
    echo   $env:OLLAMA_GPU_LAYERS = 0; ollama serve
    echo.
    pause
    exit /B 1
)
echo [OK] Ollama running

REM 2. Repo dir
echo.
echo [2/4] switching to repo dir...
cd /D "%~dp0"
if not exist "train.py" (
    echo [ERROR] train.py not found here
    pause
    exit /B 1
)
echo [OK] dir is %CD%

REM 3. Clean stale state files (optional — comment out to resume)
echo.
echo [3/4] cleaning stale event/log files (Ctrl+C to skip)...
if exist "run.log" del /Q "run.log"
if exist "activity_events.jsonl" del /Q "activity_events.jsonl"
echo [OK] cleared run.log and activity_events.jsonl

REM 4. Launch orchestrator
echo.
echo [4/4] launching v2 orchestrator...
echo.
echo ========================================
echo    LIVE: training stdout + agent events
echo ========================================
echo.
echo This terminal will print every training line and every orchestrator
echo decision in real time. To watch separately in PowerShell:
echo   Get-Content run.log -Wait -Tail 30
echo   Get-Content activity_events.jsonl -Wait -Tail 30
echo.
echo Companion tools (run in their own terminals):
echo   start_dashboard.bat         http://127.0.0.1:5052/   (read-only overview)
echo   python -m autoresearch_v2.hitl_app   http://127.0.0.1:5051/  (approve/deny)
echo.
echo Press Ctrl+C to stop.
echo ========================================
echo.

REM -u flushes Python stdout per line, matching f.flush()/print(flush=True)
REM in the Python source. This is what makes Windows cmd show live output.
python -u -m autoresearch_v2.orchestrator %*

if "%ERRORLEVEL%"=="0" (
    echo.
    echo ========================================
    echo    DONE
    echo ========================================
) else (
    echo.
    echo ========================================
    echo    EXITED (code %ERRORLEVEL%)
    echo ========================================
)

echo.
pause
