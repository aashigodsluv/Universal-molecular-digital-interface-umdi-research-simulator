@echo off
cd /d "%~dp0"
echo Starting Universal Molecular Digital Interface (UMDI) Research Simulator...
echo.
echo Batch experiments:
python run_experiments.py
echo.
echo Opening the interactive front end...
start "" "%~dp0frontend\index.html"
pause
