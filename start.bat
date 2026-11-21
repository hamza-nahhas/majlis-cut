@echo off
rem Opens the majlis_cut interface in the browser. Keep this window open while working.
cd /d "%~dp0"
set PYTHONUTF8=1
".venv\Scripts\python.exe" -m majlis_cut.ui
pause
