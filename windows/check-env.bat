@echo off
:: AEROSTRAT Pre-flight Check
:: Double-click or run from command prompt
:: Add /start to auto-launch after passing

if "%1"=="/start" (
    node "%~dp0check-env.js" --start
) else (
    node "%~dp0check-env.js"
)
pause
