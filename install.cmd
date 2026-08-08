@echo off
setlocal

where powershell >nul 2>nul
if errorlevel 1 (
    echo PowerShell not found - it ships with Windows 10/11, so this is unusual.
    exit /b 1
)

rem install.ps1 does the actual work (venv, base install, and GPU detection with a
rem CPU fallback) - kept in one place so both entry points stay in sync.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
exit /b %errorlevel%
