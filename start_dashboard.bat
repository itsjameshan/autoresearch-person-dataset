@echo off
REM ========================================
REM AutoResearch v2 Dashboard - Windows starter
REM
REM Read-only situational awareness for an in-progress orchestrator
REM run. Best CDS, recent experiments, live activity events, pending
REM HITL gates, curator data issues, Optuna sweeps. Auto-refreshes
REM every 5-30s. Default port: 5052.
REM ========================================

setlocal enabledelayedexpansion

cd /D "%~dp0"

if not exist "autoresearch_v2\dashboard.py" (
    echo [ERROR] dashboard.py not found
    pause
    exit /B 1
)

REM Defaults — override by passing extra args to this script.
set "DEFAULT_PORT=5052"
set "DEFAULT_HOST=127.0.0.1"

echo.
echo ========================================
echo    AUTORESEARCH v2 DASHBOARD
echo ========================================
echo.
echo URL:  http://%DEFAULT_HOST%:%DEFAULT_PORT%/
echo.
echo Sibling tools (run in separate terminals):
echo   - HITL UI (approve/deny gates):     http://127.0.0.1:5051/
echo   - Orchestrator main loop:           start_v2_orchestrator.bat
echo   - Live tail of training stdout:     Get-Content run.log -Wait -Tail 30
echo.
echo Press Ctrl+C to stop.
echo ========================================
echo.

python -u -m autoresearch_v2.dashboard ^
    --host %DEFAULT_HOST% ^
    --port %DEFAULT_PORT% ^
    %*

if "%ERRORLEVEL%"=="0" (
    echo.
    echo ========================================
    echo    Dashboard stopped
    echo ========================================
) else (
    echo.
    echo ========================================
    echo    Dashboard exited (code %ERRORLEVEL%)
    echo ========================================
)

echo.
pause
