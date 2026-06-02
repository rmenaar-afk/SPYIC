@echo off
:: ============================================================
:: run_scanner.bat  --  Daily Consolidation Breakout Scanner
::
:: Schedule via Windows Task Scheduler (run as admin):
::   schtasks /create /tn "BreakoutScanner" ^
::     /tr "\"C:\Users\rmena\OneDrive\Documents\Claude\Projects\Mr. Buffet\vibe-trading-skills\run_scanner.bat\"" ^
::     /sc WEEKLY /d MON,TUE,WED,THU,FRI /st 16:30 /f
:: ============================================================

cd /d "%~dp0"
set PYTHON=C:\Users\rmena\AppData\Local\Python\bin\python.exe

echo.
echo [%date% %time%] Starting Breakout Scanner...
echo.

"%PYTHON%" breakout_scanner.py

echo.
if %errorlevel% equ 0 (
    echo [%date% %time%] Scanner completed successfully.
) else (
    echo [%date% %time%] Scanner exited with error code %errorlevel%.
)
echo.
pause
