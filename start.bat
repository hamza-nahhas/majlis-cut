@echo off
rem Opens the majlis_cut interface in the browser. Keep this window open while working.
rem After an update from the interface it closes with code 3: the libraries are
rem refreshed if requirements.txt changed, and it opens again.
rem The part from :run on is one block, read whole before it runs, so an update that
rem replaces this file while it is running cannot confuse it.
cd /d "%~dp0"
set PYTHONUTF8=1
:run
(
  if not exist ".venv\Scripts\python.exe" (
    echo The tool is not set up yet. Double-click setup.bat first.
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m majlis_cut.deps
  if errorlevel 1 (
    echo Installing the libraries failed. Check the internet connection, then run start.bat again.
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m majlis_cut.ui
  if errorlevel 3 if not errorlevel 4 goto run
  pause
  exit /b
)
