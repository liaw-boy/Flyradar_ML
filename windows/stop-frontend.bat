@echo off
title AEROSTRAT Stop Frontend

echo  Stopping frontend (port 3005)...

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3005" ^| findstr "LISTENING"') do (
    echo  Stopping PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

echo  Frontend stopped. Backend still running.
timeout /t 2 /nobreak >nul
