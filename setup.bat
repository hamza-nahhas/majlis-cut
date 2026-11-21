@echo off
rem One-time setup on a new computer: makes the tool's own Python environment and
rem installs its libraries into it. start.bat refreshes them by itself after an update.
cd /d "%~dp0"
set PYTHONUTF8=1

where py >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" in its installer. Then run setup.bat again.
  pause
  exit /b 1
)
where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo WARNING: ffmpeg was not found. The tool cannot read recordings without it.
  echo See docs\INSTALL.md, step 2. Setup continues.
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment...
  py -3 -m venv .venv
  if errorlevel 1 ( echo Could not create the environment. & pause & exit /b 1 )
)
".venv\Scripts\python.exe" -m majlis_cut.deps --force
if errorlevel 1 ( echo Installing the libraries failed. Check the internet connection and run setup.bat again. & pause & exit /b 1 )

echo.
echo Setup is complete. Double-click start.bat to open the tool.
pause
