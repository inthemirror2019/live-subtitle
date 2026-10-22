@echo off
title Zhumu
cd /d "%~dp0"

set "PY=C:\Python314\python.exe"

if not exist "%PY%" (
  echo.
  echo   [!] Python not found at %PY%
  echo.
  pause
  exit /b 1
)

start "zhumu" /min "%PY%" "%~dp0zhumu.py"
exit
