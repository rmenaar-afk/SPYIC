@echo off
cd /d "%~dp0"
echo [%DATE% %TIME%] Starting 0DTE Iron Condor Trader >> logs\dte0_trader_runner.log

REM Try venv first, fall back to system python
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" dte0_trader.py
) else (
    python dte0_trader.py
)

echo [%DATE% %TIME%] Trader exited with code %ERRORLEVEL% >> logs\dte0_trader_runner.log
