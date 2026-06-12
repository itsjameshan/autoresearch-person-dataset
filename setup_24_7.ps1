# setup_24_7.ps1
# Install AutoResearch v2 as a 24/7 scheduled task on Windows.
#
# Run ONCE (as Administrator for GPU/process priority):
#   powershell -ExecutionPolicy Bypass -File setup_24_7.ps1
#
# What it does:
#   - Creates a Task Scheduler task that launches start_v2_orchestrator.bat
#   - Trigger: at system startup + repeat every 10 min (safety net)
#   - Internal watchdog handles auto-restart on orchestrator exit
#
# To remove:
#   powershell -ExecutionPolicy Bypass -File setup_24_7.ps1 -Remove

param(
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$TaskName   = "AutoResearch_v2_24_7"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatPath    = Join-Path $ScriptDir "start_24_7_scheduled.bat"
$WorkingDir = $ScriptDir

# ── Remove mode ──
if ($Remove) {
    Write-Host "Removing scheduled task: $TaskName"
    try {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "[OK] Task removed." -ForegroundColor Green
    } catch {
        Write-Host "[INFO] Task not found or already removed." -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "To stop a running orchestrator, create this file:"
    Write-Host "  $ScriptDir\autoresearch_v2\state\STOP"
    Write-Host "The watchdog will exit within ~60 seconds."
    exit 0
}

# ── Pre-checks ──
if (-not (Test-Path $BatPath)) {
    Write-Host "[ERROR] $BatPath not found" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path (Join-Path $ScriptDir "train.py"))) {
    Write-Host "[ERROR] train.py not found in $ScriptDir" -ForegroundColor Red
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  AutoResearch v2 — 24/7 Scheduler"
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Task name  : $TaskName"
Write-Host "Working dir: $WorkingDir"
Write-Host "Command    : $BatPath --scheduled"
Write-Host ""

# Remove existing task if present
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "[INFO] Removing previous instance of '$TaskName'..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Build trigger: at system startup
$trigger = New-ScheduledTaskTrigger -AtStartup

# Build action: run the 24/7 scheduled batch
$action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c `"cd /d `"$WorkingDir`" && `"$BatPath`"`"" `
    -WorkingDirectory $WorkingDir

# Build principal: current user, highest privileges
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

# Settings
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopOnIdleEnd `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -RestartCount 999 `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -Compatibility Win8

# Register
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Trigger $trigger `
        -Action $action `
        -Principal $principal `
        -Settings $settings `
        -Description "AutoResearch v2 orchestrator — 24/7 continuous training loop with built-in watchdog" `
        -Force | Out-Null

    Write-Host "[OK] Task registered successfully!" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Failed to register task: $_" -ForegroundColor Red
    exit 1
}

# Validate
$registered = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($registered) {
    Write-Host ""
    Write-Host "Task Details:" -ForegroundColor Cyan
    Write-Host "  State : $($registered.State)"
    Write-Host "  Enabled: $($registered.Settings.Enabled)"
    Write-Host ""
    Write-Host "The task will start at next system boot."
    Write-Host ""
    Write-Host "To START NOW (this terminal):" -ForegroundColor Yellow
    Write-Host "  $BatPath"
    Write-Host ""
    Write-Host "To STOP:" -ForegroundColor Yellow
    Write-Host "  Create file: $WorkingDir\autoresearch_v2\state\STOP"
    Write-Host "  The watchdog exits gracefully within ~60s."
    Write-Host ""
    Write-Host "To REMOVE schedule permanently:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File setup_24_7.ps1 -Remove"
    Write-Host ""
    Write-Host "Watchdog log: $WorkingDir\watchdog.log" -ForegroundColor DarkGray
    Write-Host ""
} else {
    Write-Host "[ERROR] Task registration verification failed" -ForegroundColor Red
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DONE — 24/7 schedule installed"
Write-Host "========================================" -ForegroundColor Cyan