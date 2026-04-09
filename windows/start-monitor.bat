REM ******************************************
REM * AEROSTRAT Claude Code Monitor Startup  *
REM ******************************************
@echo off
title Claude Code Usage Monitor (PRO)
setlocal

set "MONITOR_DIR=%~dp0Claude-Code-Usage-Monitor"
cd /d "%MONITOR_DIR%"

if not exist "venv\Scripts\claude-monitor.exe" (
    echo [ERROR] Monitor executable not found!
    pause
    exit /b 1
)

echo [INFO] Starting Claude Code Usage Monitor in PRO mode...
REM Note: --mode is not supported in this version.
.\venv\Scripts\claude-monitor.exe --plan pro --timezone Asia/Taipei --reset-hour 0
pause
