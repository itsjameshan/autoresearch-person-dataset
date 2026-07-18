@echo off
title Bilevel 训练总监视面板
echo ========================================
echo   Bilevel 训练总监视面板启动器
echo ========================================
echo.
echo 正在检查服务状态...

netstat -ano | findstr ":8080" >nul 2>&1
if %errorlevel%==0 (
    echo [OK] 监控面板服务已在运行
    echo.
) else (
    echo [启动] 正在启动监控面板服务...
    cd /d "d:\autoresearch\autoresearch_new"
    start /B python supervisor_dashboard.py
    timeout /t 3 /nobreak >nul
    echo [OK] 服务已启动
    echo.
)

echo 正在打开浏览器...
start http://127.0.0.1:8080

echo.
echo ========================================
echo   面板地址: http://127.0.0.1:8080
echo   刷新频率: 每 3 秒自动刷新
echo ========================================
echo.
pause
