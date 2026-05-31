@echo off
chcp 65001 >nul 2>&1
title 智能体总管 · AutoResearch Overseer

cd /d "d:\autoresearch\autoresearch_new"

set PYTHON=D:\Python 3.12.0\python.exe
if not exist "%PYTHON%" set PYTHON=python

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║     智 能 体 总 管                          ║
echo  ║     AutoResearch Overseer                   ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo  项目: d:\autoresearch\autoresearch_new
echo  Web UI: http://localhost:5050
echo  浏览器将自动打开...
echo.
echo  ═══════════════════════════════════════════════
echo.

"%PYTHON%" agent_overseer.py %*

echo.
echo  总管已停止。
pause