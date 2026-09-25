@echo off
cd /d "%~dp0"
echo Starting the UMDI BioSSD Python simulation engine...
start "UMDI Python Engine" cmd /k python frontend_api.py
timeout /t 2 >nul
echo Opening the interactive UMDI front end...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$f = '%~dp0frontend\index.html'; $shell = New-Object -ComObject Shell.Application; $folder = $shell.Namespace((Split-Path $f)); $item = $folder.ParseName((Split-Path $f -Leaf)); $item.InvokeVerb('openas')"
echo.
echo The browser is now connected to the Python engine at http://127.0.0.1:8766
pause
