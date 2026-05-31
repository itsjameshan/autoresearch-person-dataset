@echo off
cd /d "%~dp0"
echo ==========================================
echo   Overseer Doctor — 总管诊断医生
echo   实时监控总管 Web UI 日志并自动修复错误
echo ==========================================
echo.
python -u -m autoresearch_v2.agents.overseer_doctor --watch --interval 15
pause