@echo off
title AEROSTRAT Stop All

echo.
echo  Stopping AEROSTRAT (ports 3000 and 5173)...
echo.

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3000" ^| findstr "LISTENING"') do (
    echo  Stopping backend PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":3005" ^| findstr "LISTENING"') do (
    echo  Stopping frontend PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo  Done.
timeout /t 2 /nobreak >nul
