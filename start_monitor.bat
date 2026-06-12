@echo off
REM start_monitor.bat — 启动 AutoResearch 持续监督智能体
REM 用法:
REM   start_monitor.bat             一次性生成 handoff.md + blog.md
REM   start_monitor.bat --watch      持续监督模式
REM   start_monitor.bat --blog-only  只生成 blog.md
REM   start_monitor.bat -h           查看帮助

cd /d "%~dp0"

python monitor_agent.py %*
pause