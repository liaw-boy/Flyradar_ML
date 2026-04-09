@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo.
echo  ====================================================
echo  AEROSTRAT - First-time Setup
echo  ====================================================
echo.

:: ── Check Node.js ────────────────────────────────────────
where node >nul 2>&1
if errorlevel 1 (
    echo  [FAIL] Node.js not found.
    echo         Please install v24+ from https://nodejs.org
    pause & exit /b 1
)
for /f %%v in ('node -v') do echo  [OK] Node.js %%v

:: ── Check .env ────────────────────────────────────────────
if not exist "backend\.env" (
    if exist "backend\.env.example" (
        copy "backend\.env.example" "backend\.env" >nul
        echo  [WARN] backend\.env automatically created from example.
        echo         Please remember to configure your real credentials later!
    ) else (
        echo  [FAIL] backend\.env not found
        echo         Copy your .env file to backend\.env first.
        pause & exit /b 1
    )
) else (
    echo  [OK] .env found
)

if not exist "client\.env" (
    if exist "client\.env.example" (
        copy "client\.env.example" "client\.env" >nul
        echo  [WARN] client\.env automatically created from example.
    )
)

:: ── npm install backend (needed before MongoDB check) ─────
echo.
echo  [1/6] Installing backend dependencies...
cd backend
call npm install --silent
if errorlevel 1 ( echo  [FAIL] backend npm install failed & pause & exit /b 1 )
echo  [OK] Backend dependencies installed
cd ..

:: ── Check MongoDB (TCP port check, no external modules) ───
echo  Checking MongoDB...
node -e "const n=require('net');const s=n.createConnection({port:27017,host:'127.0.0.1'},()=>{s.destroy();process.exit(0)});s.on('error',()=>process.exit(1));setTimeout(()=>process.exit(1),3000);" >nul 2>&1
if errorlevel 1 (
    echo  [FAIL] MongoDB not running on localhost:27017
    echo         Please start MongoDB ^(run as Admin: net start MongoDB^)
    pause & exit /b 1
)
echo  [OK] MongoDB connected

:: ── npm install client ────────────────────────────────────
echo  [2/6] Installing client dependencies...
cd client
call npm install --silent
if errorlevel 1 ( echo  [FAIL] client npm install failed & pause & exit /b 1 )
echo  [OK] Client dependencies installed
cd ..

:: ── Clone AircraftShapesSVG ───────────────────────────────
echo  [3/6] Setting up AircraftShapesSVG...
if not exist "assets\AircraftShapesSVG\Shapes SVG" (
    if not exist "assets" mkdir assets
    if exist "AircraftShapesSVG" (
        move "AircraftShapesSVG" "assets\AircraftShapesSVG" >nul 2>&1
    )
    if not exist "assets\AircraftShapesSVG" (
        git clone https://github.com/RexKramer1/AircraftShapesSVG.git assets\AircraftShapesSVG >nul 2>&1
        if errorlevel 1 ( echo  [WARN] Failed to clone AircraftShapesSVG ^(Git not installed?^). Missing SVG icons. )
    )
    echo  [OK] AircraftShapesSVG ready
) else (
    echo  [OK] AircraftShapesSVG already exists
)

:: ── syncOsintData ─────────────────────────────────────────
echo  [4/6] Syncing route and airport data (~598k records, takes 2-3 min)...
cd backend
node scripts/syncOsintData.js
if errorlevel 1 ( echo  [WARN] syncOsintData had errors, continuing... )
echo  [OK] Route/airport data synced
cd ..

:: ── seed-shapes ───────────────────────────────────────────
echo  [5/6] Seeding aircraft shapes...
cd backend
call npm run seed-shapes
if errorlevel 1 ( echo  [WARN] seed-shapes had errors, continuing... )
echo  [OK] Aircraft shapes seeded
cd ..

:: ── Copy SVG icons to backend/public/svg ──────────────────
echo  Copying SVG icons to backend/public/svg...
if not exist "backend\public\svg" mkdir "backend\public\svg"
if exist "assets\AircraftShapesSVG\Shapes SVG" (
    xcopy "assets\AircraftShapesSVG\Shapes SVG\*.svg" "backend\public\svg\" /Y /Q >nul 2>&1
    echo  [OK] SVG icons copied
) else (
    echo  [WARN] AircraftShapesSVG not found, skipping SVG copy
)

:: ── sync-mictronics ───────────────────────────────────────
echo  [6/6] Syncing Mictronics aircraft database (~530k records, takes 3-5 min)...
cd backend
call npm run sync-mictronics
if errorlevel 1 ( echo  [WARN] sync-mictronics had errors, continuing... )
echo  [OK] Mictronics sync done
cd ..

:: ── Final check ───────────────────────────────────────────
echo.
echo  ====================================================
echo  Running final environment check...
echo  ====================================================
node check-env.js

echo.
echo  ====================================================
echo  Setup complete! To start the system:
echo    Backend:  cd backend ^&^& npm start
echo    Frontend: cd client  ^&^& npm run dev
echo.
echo  Or run:  node check-env.js --start
echo  ====================================================
echo.
pause
