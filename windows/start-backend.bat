@echo off
title AEROSTRAT Backend (Port 3000)
cd /d "%~dp0backend"

echo.
echo  ==========================================
echo  AEROSTRAT Backend Engine v5.0.0
echo  Port 3000  /  MongoDB Local
echo  ==========================================
echo.

if not exist ".env" (
    echo  [ERROR] backend\.env not found!
    echo  Please create backend\.env with MONGODB_URI, PORT, OPENSKY credentials.
    echo.
    pause
    exit /b 1
)

if not exist "node_modules" (
    echo  [SETUP] Installing dependencies...
    call npm install
    echo.
)

echo  [INFO] Health check: http://localhost:3000/api/health
echo  [INFO] Press Ctrl+C to stop backend
echo.

node server.js

echo.
echo  Backend stopped.
pause
