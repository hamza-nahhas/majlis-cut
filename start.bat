@echo off
rem Opens the darsmontage interface in the browser. Keep this window open while working.
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m darsmontage.ui
pause
