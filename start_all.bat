@echo off
REM ========================================
REM  AutoResearch v2 — One-Click Launcher
REM
REM  Double-click this file to launch:
REM    1. Ollama service (if installed)
REM    2. v2 Orchestrator (live training + agent events)
REM    3. Web Dashboard  (http://127.0.0.1:5052/)
REM    4. Live log tail  (run.log streaming)
REM
REM  Each component runs in its own window.
REM  Close them individually or Ctrl+C.
REM ========================================

setlocal enabledelayedexpansion

cd /D "%~dp0"

echo.
echo ========================================
echo    AUTORESEARCH v2 — LAUNCHING ALL
echo ========================================
echo.

REM ── Terminal 1: Ollama Service ──────────────────────────────────
echo [1/4] Ollama service...

ollama --version >NUL 2>&1
if "%ERRORLEVEL%"=="1" (
    echo        [WARN] Ollama is not installed.
    echo        LLM agents will NOT work. Continuing without Ollama...
    echo        To install: https://ollama.com
    goto :ollama_done
)

REM Check if Ollama is already running
ollama list >NUL 2>&1
if "%ERRORLEVEL%"=="0" (
    echo        Ollama is already running — reusing existing instance.
    goto :ollama_ready
)

echo        Starting Ollama in a new terminal...
start "Ollama Service" cmd /k ^
    "echo Starting Ollama... ^&^& ^
     echo. ^&^& ^
     ollama serve"

REM Wait up to 60 seconds for Ollama to become ready
echo        Waiting for Ollama to start...
set "tries=0"
:wait_ollama
timeout /T 2 /NOBREAK >NUL
ollama list >NUL 2>&1
if "%ERRORLEVEL%"=="0" (
    echo        Ollama is ready!
    goto :ollama_ready
)
set /a tries+=1
if !tries! LSS 30 goto :wait_ollama
echo.
echo        [WARN] Ollama did not start within 60 seconds.
echo        LLM agents may fail. Continuing anyway...
goto :ollama_done

:ollama_ready
REM Make sure gemma3:4b is pulled
ollama list | findstr "gemma3:4b" >NUL
if "%ERRORLEVEL%"=="1" (
    echo        Pulling gemma3:4b (this may take a few minutes)...
    ollama pull gemma3:4b
    if "%ERRORLEVEL%"=="1" (
        echo        [WARN] Could not pull gemma3:4b. LLM agents may fail.
    )
)
:ollama_done

REM ── Terminal 2: Orchestrator ────────────────────────────────────
echo.
echo [2/4] Launching orchestrator (main loop)...
start "Autoresearch v2 Orchestrator" cmd /k ^
    "cd /D \"%~dp0\" ^&^& ^
     echo ======================================== ^&^& ^
     echo    AUTORESEARCH v2 - MULTI AGENT ^&^& ^
     echo ======================================== ^&^& ^
     echo. ^&^& ^
     echo LIVE: training stdout + agent events ^&^& ^
     echo Dashboard: http://127.0.0.1:5052/ ^&^& ^
     echo Ctrl+C to stop. ^&^& ^
     echo ======================================== ^&^& ^
     echo. ^&^& ^
     set PYTHONIOENCODING=utf-8 ^&^& ^
     set PYTHONUTF8=1 ^&^& ^
     python -u -m autoresearch_v2.orchestrator %* ^&^& ^
     echo. ^&^& ^
     echo Orchestrator finished. Close this window when ready. ^&^& ^
     pause >NUL"

REM Brief pause so the orchestrator starts creating files
timeout /T 3 /NOBREAK >NUL

REM ── Terminal 3: Dashboard ───────────────────────────────────────
echo.
echo [3/4] Launching web dashboard...
start "Autoresearch v2 Dashboard" cmd /k ^
    "cd /D \"%~dp0\" ^&^& ^
     echo ======================================== ^&^& ^
     echo    AUTORESEARCH v2 DASHBOARD ^&^& ^
     echo ======================================== ^&^& ^
     echo. ^&^& ^
     echo URL:  http://127.0.0.1:5052/ ^&^& ^
     echo Ctrl+C to stop. ^&^& ^
     echo ======================================== ^&^& ^
     echo. ^&^& ^
     python -u -m autoresearch_v2.dashboard ^
         --host 127.0.0.1 --port 5052 ^&^& ^
     echo. ^&^& ^
     echo Dashboard stopped. Close this window when ready. ^&^& ^
     pause >NUL"

REM ── Terminal 4: Live Log Tail ────────────────────────────────────
echo.
echo [4/4] Launching live log tail...

timeout /T 5 /NOBREAK >NUL

start "Live Training Log (run.log)" powershell -NoExit -Command ^
    "Write-Host '========================================' -ForegroundColor Cyan; ^
     Write-Host '   LIVE TRAINING LOG (run.log)' -ForegroundColor Cyan; ^
     Write-Host '========================================' -ForegroundColor Cyan; ^
     Write-Host ''; ^
     Write-Host 'Streaming training output in real time.'; ^
     Write-Host 'Close this window or press Ctrl+C to stop.'; ^
     Write-Host '========================================'; ^
     Write-Host ''; ^
     while (-not (Test-Path 'run.log')) { ^
         Write-Host 'Waiting for run.log...'; ^
         Start-Sleep -Seconds 3 ^
     }; ^
     Get-Content run.log -Wait -Tail 30"

REM ── Done ─────────────────────────────────────────────────────────
echo.
echo ========================================
echo    LAUNCH COMPLETE
echo ========================================
echo.
echo   Terminal 1: Ollama Service
echo   Terminal 2: v2 Orchestrator (main loop)
echo   Terminal 3: Dashboard  http://127.0.0.1:5052/
echo   Terminal 4: Live log  (run.log tail)
echo.
echo ========================================
echo.

REM Auto-open dashboard in browser
start http://127.0.0.1:5052/

echo.
echo Press any key to close this window (terminals stay open).
pause >NUL