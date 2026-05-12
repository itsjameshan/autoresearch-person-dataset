@echo off
REM ========================================
REM AutoResearch v2 HITL App - Windows starter
REM
REM Approve/deny HITL gates from a dedicated UI.
REM Default URL: http://127.0.0.1:5051/
REM ========================================

setlocal enabledelayedexpansion

cd /D "%~dp0"

if not exist "autoresearch_v2\hitl_app.py" (
    echo [ERROR] hitl_app.py not found
    pause
    exit /B 1
)

REM Defaults - override by passing extra args to this script.
set "DEFAULT_PORT=5051"
set "DEFAULT_HOST=127.0.0.1"

echo.
echo ========================================
echo    AUTORESEARCH v2 HITL APP
echo ========================================
echo.
echo URL:  http://%DEFAULT_HOST%:%DEFAULT_PORT%/
echo.
echo Sibling tools (run in separate terminals):
echo   - Dashboard (read-only):             http://127.0.0.1:5052/
echo   - Orchestrator main loop:            start_v2_orchestrator.bat
echo.
echo Press Ctrl+C to stop.
echo ========================================
echo.

python -u -m autoresearch_v2.hitl_app ^
    --host %DEFAULT_HOST% ^
    --port %DEFAULT_PORT% ^
    %*

if "%ERRORLEVEL%"=="0" (
    echo.
    echo ========================================
    echo    HITL App stopped
    echo ========================================
) else (
    echo.
    echo ========================================
    echo    HITL App exited (code %ERRORLEVEL%)
    echo ========================================
)

echo.
pause
