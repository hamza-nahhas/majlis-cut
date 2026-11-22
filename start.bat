@echo off
rem Opens the majlis_cut interface in the browser. Keep this window open while working.
rem Uses the Python and ffmpeg that the installer puts next to the tool (python\,
rem ffmpeg\bin\); a copy set up with setup.bat uses .venv and the system ffmpeg.
rem After an update from the interface it closes with code 3: the libraries are
rem refreshed if requirements.txt changed, and it opens again.
rem The part from :run on is one block, read whole before it runs, so an update that
rem replaces this file while it is running cannot confuse it.
cd /d "%~dp0"
set PYTHONUTF8=1
set "PY=.venv\Scripts\python.exe"
rem The bundled Python must not pick up libraries installed elsewhere on this machine.
if exist "python\python.exe" (
  set "PY=python\python.exe"
  set PYTHONNOUSERSITE=1
  set PYTHONPATH=
  set PYTHONHOME=
)
if exist "ffmpeg\bin\ffmpeg.exe" set "PATH=%~dp0ffmpeg\bin;%PATH%"
:run
(
  if not exist "%PY%" (
    echo The tool is not set up yet. Double-click setup.bat first.
    pause
    exit /b 1
  )
  "%PY%" -m majlis_cut.deps
  if errorlevel 1 (
    echo Installing the libraries failed. Check the internet connection, then run start.bat again.
    pause
    exit /b 1
  )
  "%PY%" -m majlis_cut.ui
  if errorlevel 3 if not errorlevel 4 goto run
  pause
  exit /b
)
