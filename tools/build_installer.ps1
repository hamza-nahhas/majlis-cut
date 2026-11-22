# Builds the Windows installer: build\majlis-cut-setup-v<version>.exe
#
#   powershell -ExecutionPolicy Bypass -File tools\build_installer.ps1 [-Iscc <path to ISCC.exe>]
#
# Needs git, uv (https://docs.astral.sh/uv/) and Inno Setup 7. It packs the
# committed HEAD (like the release zip), a standalone Python 3.13 with the
# libraries from requirements.txt already installed, and ffmpeg/ffprobe.
param([string]$Iscc = "")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"            # Invoke-WebRequest is very slow with it
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$version = (Select-String -Path "majlis_cut\__init__.py" -Pattern '__version__ = "([^"]+)"').Matches[0].Groups[1].Value
$build = Join-Path $root "build"
$app = Join-Path $build "app"
if (Test-Path $build) { Remove-Item -Recurse -Force $build }
New-Item -ItemType Directory -Force $app | Out-Null
Write-Host "== majlis-cut $version"

# 1. The tool's own files, exactly as the release zip has them.
git archive --format=zip -o "$build\app.zip" HEAD
if ($LASTEXITCODE) { throw "git archive failed" }
Expand-Archive "$build\app.zip" $app

# 2. A standalone Python (python-build-standalone, the one uv uses): relocatable,
#    with tkinter for the file pickers. Kept out of the registry and PATH.
$env:UV_PYTHON_INSTALL_DIR = "$build\pythons"
$env:UV_PYTHON_INSTALL_REGISTRY = "0"
$env:UV_PYTHON_INSTALL_BIN = "0"
uv python install 3.13
if ($LASTEXITCODE) { throw "uv python install failed" }
$py = Get-ChildItem "$build\pythons" -Directory | Where-Object Name -like "cpython-3.13*" | Select-Object -First 1
Copy-Item -Recurse $py.FullName "$app\python"
Remove-Item "$app\python\Lib\EXTERNALLY-MANAGED" -ErrorAction SilentlyContinue   # uv's marker; pip must work here

# Libraries go into that Python itself. User-wide packages must not count as installed.
$env:PYTHONNOUSERSITE = "1"
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Push-Location $app
& "$app\python\python.exe" -m majlis_cut.deps --force
$code = $LASTEXITCODE
Pop-Location
if ($code) { throw "installing the libraries failed" }
& "$app\python\python.exe" -c "import numpy, scipy, fitz, faster_whisper, tkinter, winrt.windows.media.ocr; print('libraries ok')"
if ($LASTEXITCODE) { throw "the bundled Python cannot load the libraries" }

# 3. ffmpeg and ffprobe: the newest release branch of BtbN's GPL builds (GPL for libx264).
$headers = @{}
if ($env:GH_TOKEN) { $headers.Authorization = "Bearer $env:GH_TOKEN" }   # CI runners share the anonymous limit
$rel = Invoke-RestMethod "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/latest" -Headers $headers
$asset = $rel.assets | Where-Object { $_.name -match '^ffmpeg-n(\d+)\.(\d+)-latest-win64-gpl-\1\.\2\.zip$' } |
    Sort-Object { [version]($_.name -replace '^ffmpeg-n([\d.]+)-.*$', '$1') } -Descending | Select-Object -First 1
if (-not $asset) { throw "no ffmpeg build found" }
Write-Host "== $($asset.name)"
Invoke-WebRequest $asset.browser_download_url -OutFile "$build\ffmpeg.zip"
Expand-Archive "$build\ffmpeg.zip" "$build\ffmpeg"
$ff = Get-ChildItem "$build\ffmpeg" -Directory | Select-Object -First 1
New-Item -ItemType Directory -Force "$app\ffmpeg\bin" | Out-Null
Copy-Item "$($ff.FullName)\bin\ffmpeg.exe", "$($ff.FullName)\bin\ffprobe.exe" "$app\ffmpeg\bin"
Copy-Item "$($ff.FullName)\LICENSE.txt" "$app\ffmpeg\LICENSE.txt"
Set-Content "$app\ffmpeg\SOURCE.txt" "ffmpeg built by https://github.com/BtbN/FFmpeg-Builds ($($asset.name)), GPL; source: https://ffmpeg.org/download.html"

# 4. The installer.
if (-not $Iscc) {
    $Iscc = @("$env:ProgramFiles\Inno Setup 7\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe",
              "$env:LOCALAPPDATA\Programs\Inno Setup 7\ISCC.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Iscc) { throw "Inno Setup 7 not found; pass -Iscc <path to ISCC.exe>" }
& $Iscc "/DMyVersion=$version" "/DSourceDir=$app" "/O$build" "installer\majlis-cut.iss"
if ($LASTEXITCODE) { throw "ISCC failed" }
Get-Item "$build\majlis-cut-setup-v$version.exe" | ForEach-Object { "{0}  {1:N0} MB" -f $_.Name, ($_.Length / 1MB) }
