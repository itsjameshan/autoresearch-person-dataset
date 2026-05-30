@echo off
REM ========================================
REM  AutoResearch v2 — Continuous Watchdog
REM
REM  Self-restarting orchestrator loop.
REM  When the orchestrator finishes (max iters,
REM  stagnation, target met), it auto-restarts.
REM
REM  To stop:  Ctrl+C  twice, or create file:
REM    autoresearch_v2\state\STOP
REM ========================================

setlocal enabledelayedexpansion

REM ── Config ──
set MAX_RESTARTS=50
set RESTART_DELAY=60
set CRASH_THRESHOLD=3
set /a crash_count=0
set /a restart_count=0
set FIRST_RUN=1

echo.
echo ========================================
echo    AUTORESEARCH v2 — CONTINUOUS MODE
echo ========================================
echo.
echo The orchestrator will auto-restart on exit.
echo.
echo Stop methods:
echo   - Ctrl+C (twice)
echo   - Create file: autoresearch_v2\state\STOP
echo.
echo ========================================
echo.

REM 1. Ollama check
echo [CHECK] Ollama...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    echo.
    echo [WARN] Ollama is not running!
    echo Start it:  ollama serve
    echo.
    pause
    exit /B 1
)
echo [OK] Ollama running

REM 2. Repo dir
cd /D "%~dp0"
if not exist "train.py" (
    echo [ERROR] train.py not found in %CD%
    pause
    exit /B 1
)
echo [OK] Working dir: %CD%

REM 3. UTF-8 console
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM ── Watchdog Loop ──
:watchdog_loop

REM Stop signal check
if exist "autoresearch_v2\state\STOP" (
    echo.
    echo [STOP] Manual stop signal detected. Exiting.
    del /Q "autoresearch_v2\state\STOP"
    goto :done
)

REM Only clear stale logs on the very first launch
if "!FIRST_RUN!"=="1" (
    echo.
    echo [CLEAN] First launch — clearing stale logs...
    if exist "run.log"            del /Q "run.log"
    if exist "activity_events.jsonl" del /Q "activity_events.jsonl"
    set FIRST_RUN=0
)

echo.
echo ========================================
echo    LAUNCH  #!restart_count!  (crash streak: !crash_count!)
echo ========================================
echo.

python -u -m autoresearch_v2.orchestrator %*
set EXIT_CODE=%ERRORLEVEL%

set /a restart_count+=1

echo.
echo ========================================
echo    ORCHESTRATOR EXITED  (code !EXIT_CODE!)
echo    Restarts: !restart_count!
echo ========================================

REM Crash tracking
if !EXIT_CODE! NEQ 0 (
    set /a crash_count+=1
    echo    Consecutive failures: !crash_count!
) else (
    set /a crash_count=0
)

REM Safety bail — too many consecutive crashes
if !crash_count! GEQ %CRASH_THRESHOLD% (
    echo.
    echo [CRIT] !crash_count! consecutive failures.
    echo Stopping for safety. Check logs before restarting.
    goto :done
)

REM Stop signal check
if exist "autoresearch_v2\state\STOP" (
    echo.
    echo [STOP] Manual stop signal detected. Exiting.
    del /Q "autoresearch_v2\state\STOP"
    goto :done
)

REM Max restarts
if !restart_count! GEQ %MAX_RESTARTS% (
    echo.
    echo [INFO] Max restarts reached (!MAX_RESTARTS!). Exiting.
    goto :done
)

echo.
echo [WAIT] Restarting in %RESTART_DELAY%s ...
timeout /T %RESTART_DELAY% /NOBREAK >NUL

goto :watchdog_loop

:done
echo.
echo ========================================
echo    WATCHDOG STOPPED
echo    Total restarts: !restart_count!
echo ========================================
echo.
pause