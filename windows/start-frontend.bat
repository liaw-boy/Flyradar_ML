@echo off
title AEROSTRAT Frontend (Port 3005)
cd /d "%~dp0client"

echo.
echo  ==========================================
echo  AEROSTRAT Frontend UI v5.0.0

echo  Port 3005  /  Proxy to localhost:3000
echo  ==========================================
echo.

if not exist "node_modules" (
    echo  [SETUP] Installing dependencies...
    call npm install
    echo.
)

echo  [INFO] Open browser: http://localhost:3005
echo  [INFO] Make sure backend is running first!
echo  [INFO] Press Ctrl+C to stop frontend only
echo.

npm run dev

echo.
echo  Frontend stopped. Backend is still running.
pause
