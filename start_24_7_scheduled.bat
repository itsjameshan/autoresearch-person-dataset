@echo off
chcp 65001 >NUL 2>&1
REM ============================================================
REM  AutoResearch v2 - 24/7 Scheduler Launcher
REM
REM  Self-restarting orchestrator loop for Task Scheduler.
REM  Logs: watchdog.log / run.log / activity_events.jsonl
REM
REM  Stop: create file  autoresearch_v2\state\STOP
REM ============================================================

setlocal enabledelayedexpansion

REM ---- Config ----
set MAX_RESTARTS=9999
set RESTART_DELAY=60
set CRASH_THRESHOLD=5
set /a crash_count=0
set /a restart_count=0
set FIRST_RUN=1
set LOGFILE=%~dp0watchdog.log

REM ---- Activate conda env ----
call conda activate TraeAI-3 2>NUL
if errorlevel 1 (
    echo [WARN] conda activate TraeAI-3 failed - using system python
)

REM ---- Repo dir ----
cd /D "%~dp0"

echo [%date% %time%] === Watchdog started (24/7) === >> "%LOGFILE%"
echo [%date% %time%] Dir: %CD% >> "%LOGFILE%"

echo.
echo ============================================================
echo    AUTORESEARCH v2 - 24/7 WATCHDOG
echo ============================================================
echo.
echo LOGS:   watchdog.log / run.log / activity_events.jsonl
echo STOP:   create autoresearch_v2\state\STOP
echo ============================================================
echo.

REM ---- Wait for Ollama ----
echo [WAIT] Allowing 120s for Ollama to start...
echo [%date% %time%] Waiting for Ollama (max 120s) >> "%LOGFILE%"
set OLLAMA_READY=0
for /L %%i in (1,1,24) do (
    tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
    if "!ERRORLEVEL!"=="0" (
        set OLLAMA_READY=1
        goto :ollama_ok
    )
    timeout /T 5 /NOBREAK >NUL
)
:ollama_ok
if "!OLLAMA_READY!"=="0" (
    echo [WARN] Ollama not detected after 120s. Will retry in loop.
    echo [%date% %time%] WARN: Ollama not found after 120s >> "%LOGFILE%"
) else (
    echo [OK] Ollama running
    echo [%date% %time%] Ollama OK >> "%LOGFILE%"
)

REM ---- Pre-flight check ----
if not exist "train.py" (
    echo [ERROR] train.py not found in %CD%
    echo [%date% %time%] ERROR: train.py not found >> "%LOGFILE%"
    exit /B 1
)
echo [OK] Working dir: %CD%

where python >NUL 2>&1
if errorlevel 1 (
    echo [ERROR] python not found in PATH
    echo [%date% %time%] ERROR: python not found >> "%LOGFILE%"
    exit /B 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(sys.executable)"') do set PYTHON_EXE=%%i
echo [OK] Python: !PYTHON_EXE!
echo [%date% %time%] Python: !PYTHON_EXE! >> "%LOGFILE%"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM ============================================================
REM  WATCHDOG LOOP
REM ============================================================
:watchdog_loop

REM Stop signal check
if exist "autoresearch_v2\state\STOP" (
    echo.
    echo [STOP] STOP file detected. Exiting gracefully.
    echo [%date% %time%] STOP signal - exiting >> "%LOGFILE%"
    del /Q "autoresearch_v2\state\STOP"
    goto :done
)

REM Clear stale logs only on first launch
if "!FIRST_RUN!"=="1" (
    echo.
    echo [CLEAN] First launch - clearing stale logs...
    if exist "run.log"            del /Q "run.log"
    if exist "activity_events.jsonl" del /Q "activity_events.jsonl"
    echo [%date% %time%] Logs cleared (first launch) >> "%LOGFILE%"
    set FIRST_RUN=0
)

REM Ollama alive check (every loop)
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "!ERRORLEVEL!"=="1" (
    echo.
    echo [RETRY] Ollama down. Waiting 30s...
    echo [%date% %time%] Ollama down - retry in 30s >> "%LOGFILE%"
    timeout /T 30 /NOBREAK >NUL
    goto :watchdog_loop
)

echo.
echo ============================================================
echo    LAUNCH #!restart_count!  (crash streak: !crash_count!)
echo ============================================================
echo [%date% %time%] Orchestrator launch #!restart_count! >> "%LOGFILE%"
echo.

python -u -m autoresearch_v2.orchestrator %*
set EXIT_CODE=%ERRORLEVEL%

set /a restart_count+=1

echo.
echo ============================================================
echo    ORCHESTRATOR EXITED  (code !EXIT_CODE!)
echo    Total restarts: !restart_count!
echo ============================================================
echo [%date% %time%] Exit code=!EXIT_CODE!  restart #!restart_count! >> "%LOGFILE%"

REM Crash tracking
if !EXIT_CODE! NEQ 0 (
    set /a crash_count+=1
    echo    Crash count: !crash_count!
) else (
    set /a crash_count=0
)

REM Consecutive crash bail
if !crash_count! GEQ %CRASH_THRESHOLD% (
    echo.
    echo [CRIT] !crash_count! consecutive crashes. Stopping.
    echo [%date% %time%] CRIT: !crash_count! crashes - STOP >> "%LOGFILE%"
    goto :done
)

REM Stop signal check
if exist "autoresearch_v2\state\STOP" (
    echo.
    echo [STOP] STOP file detected. Exiting.
    echo [%date% %time%] STOP signal - exiting >> "%LOGFILE%"
    del /Q "autoresearch_v2\state\STOP"
    goto :done
)

REM Max restarts
if !restart_count! GEQ %MAX_RESTARTS% (
    echo.
    echo [INFO] Max restarts reached. Exiting.
    goto :done
)

echo.
echo [WAIT] Cooling down %RESTART_DELAY%s before next launch...
timeout /T %RESTART_DELAY% /NOBREAK >NUL

goto :watchdog_loop

REM ============================================================
:done
echo.
echo ============================================================
echo    WATCHDOG STOPPED
echo    Total restarts: !restart_count!
echo ============================================================
echo [%date% %time%] Watchdog stopped. Restarts: !restart_count! >> "%LOGFILE%"
exit /B 0