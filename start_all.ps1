# ========================================
#  AutoResearch v2 — One-Click Launcher
#  PowerShell version (reliable on Chinese Windows)
# ========================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host ''
Write-Host '========================================'
Write-Host '   AUTORESEARCH v2 — LAUNCHING ALL'
Write-Host '========================================'
Write-Host ''

# ── Check if already running ────────────────────────────────────
$dashboardUrl = 'http://127.0.0.1:5052/'
$overseerUrl  = 'http://127.0.0.1:5050/'

Write-Host 'Checking if AutoResearch is already running...'
$alreadyRunning = $false
try {
    $r = Invoke-WebRequest -Uri "$dashboardUrl/api/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
    if ($r.StatusCode -eq 200) { $alreadyRunning = $true }
} catch {
    try {
        $r = Invoke-WebRequest -Uri "$overseerUrl/api/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            $alreadyRunning = $true
            $dashboardUrl = $overseerUrl
        }
    } catch {}
}

if ($alreadyRunning) {
    Write-Host 'AutoResearch is already running!'
    Write-Host "Opening $dashboardUrl ..."
    Start-Process $dashboardUrl
    Write-Host ''
    Write-Host 'Press any key to close this window.'
    $null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
    exit 0
}

Write-Host 'No existing instance detected. Starting fresh...'
Write-Host ''

# ── Terminal 1: Ollama Service ──────────────────────────────────
Write-Host '[1/4] Ollama service...'

$ollamaInstalled = $false
$ollamaRunning   = $false

& { ollama --version 2>$null }
if ($LASTEXITCODE -eq 0) {
    $ollamaInstalled = $true
    & { ollama list 2>$null }
    if ($LASTEXITCODE -eq 0) {
        $ollamaRunning = $true
    }
}

if (-not $ollamaInstalled) {
    Write-Host '       [WARN] Ollama is not installed.'
    Write-Host '       LLM agents will NOT work. Continuing without Ollama...'
    Write-Host '       To install: https://ollama.com'
}
elseif ($ollamaRunning) {
    Write-Host '       Ollama is already running - reusing existing instance.'
}
else {
    Write-Host '       Starting Ollama in a new terminal...'
    Start-Process cmd.exe -ArgumentList '/k', '_launch_ollama.bat' -WorkingDirectory $ScriptDir

    Write-Host '       Waiting for Ollama to start...'
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        & { ollama list 2>$null }
        if ($LASTEXITCODE -eq 0) {
            Write-Host '       Ollama is ready!'
            $ready = $true
            break
        }
    }
    if (-not $ready) {
        Write-Host ''
        Write-Host '       [WARN] Ollama did not start within 60 seconds.'
        Write-Host '       LLM agents may fail. Continuing anyway...'
    }
    $ollamaRunning = $ready
}

# Make sure gemma3:4b is pulled
if ($ollamaRunning) {
    $models = & { ollama list 2>$null }
    if ($models -notmatch 'gemma3:4b') {
        Write-Host '       Pulling gemma3:4b (this may take a few minutes)...'
        & { ollama pull gemma3:4b }
        if ($LASTEXITCODE -ne 0) {
            Write-Host '       [WARN] Could not pull gemma3:4b. LLM agents may fail.'
        }
    }
}

# ── Terminal 2: Orchestrator ────────────────────────────────────
Write-Host ''
Write-Host '[2/4] Launching orchestrator (main loop)...'
Start-Process cmd.exe -ArgumentList '/k', '_launch_orchestrator.bat' -WorkingDirectory $ScriptDir

Start-Sleep -Seconds 3

# ── Terminal 3: Dashboard ───────────────────────────────────────
Write-Host ''
Write-Host '[3/4] Launching web dashboard...'
Start-Process cmd.exe -ArgumentList '/k', '_launch_dashboard.bat' -WorkingDirectory $ScriptDir

# ── Terminal 4: Live Log Tail ────────────────────────────────────
Write-Host ''
Write-Host '[4/4] Launching live log tail...'

Start-Sleep -Seconds 5

Start-Process powershell.exe -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', '_launch_log.ps1' -WorkingDirectory $ScriptDir

# ── Done ─────────────────────────────────────────────────────────
Write-Host ''
Write-Host '========================================'
Write-Host '   LAUNCH COMPLETE'
Write-Host '========================================'
Write-Host ''
Write-Host '  Terminal 1: Ollama Service'
Write-Host '  Terminal 2: v2 Orchestrator (main loop)'
Write-Host '  Terminal 3: Dashboard  http://127.0.0.1:5052/'
Write-Host '  Terminal 4: Live log  (run.log tail)'
Write-Host ''
Write-Host '========================================'

# ── Auto-open dashboard in browser (wait for it to be ready) ────
Write-Host ''
Write-Host 'Waiting for web UI to be ready...'

$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        # Check dashboard first, then overseer
        $r = Invoke-WebRequest -Uri "$dashboardUrl/api/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        try {
            $r = Invoke-WebRequest -Uri "$overseerUrl/api/health" -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
            if ($r.StatusCode -eq 200) {
                $dashboardUrl = $overseerUrl
                $ready = $true
                break
            }
        } catch {}
    }
    Start-Sleep -Seconds 2
}

if ($ready) {
    Write-Host "Opening $dashboardUrl ..."
    Start-Process $dashboardUrl
    Write-Host 'Browser opened!'
} else {
    Write-Host '[WARN] Web UI not ready after 60s. Try manually:'
    Write-Host "       Overseer UI : $overseerUrl"
    Write-Host "       Dashboard   : $dashboardUrl"
    Write-Host ''
    Write-Host 'Opening browser anyway...'
    Start-Process $dashboardUrl
}

Write-Host ''
Write-Host 'Press any key to close this window (terminals stay open).'
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')