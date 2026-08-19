@echo off
title Stock Prediction System
cd /d "%~dp0"

echo ==========================================
echo   Stock Prediction - ARIMA/LSTM/SVR/BP
echo ==========================================
echo.

REM ---- auto-select Python (anaconda first, then PATH) ----
set "PY_CMD="
if exist "D:\anacoda\python.exe" set "PY_CMD=D:\anacoda\python.exe"
if not defined PY_CMD if exist "C:\ProgramData\Anaconda3\python.exe" set "PY_CMD=C:\ProgramData\Anaconda3\python.exe"
if not defined PY_CMD if exist "C:\ProgramData\anaconda3\python.exe" set "PY_CMD=C:\ProgramData\anaconda3\python.exe"
if not defined PY_CMD if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PY_CMD if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PY_CMD set "PY_CMD=python"

echo [Python] using: %PY_CMD%
"%PY_CMD%" --version
echo.

REM ---- check & install dependencies ----
"%PY_CMD%" -c "import fastapi, uvicorn, tensorflow, pmdarima, arch" >nul 2>&1
if errorlevel 1 (
    echo [WARN] dependencies missing, installing...
    "%PY_CMD%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] dependency install failed.
        echo  1. Python version must be 3.8-3.10
        echo  2. check network and pip source
        echo.
        pause
        exit /b 1
    )
)

echo.
echo [START] open http://127.0.0.1:8000 in browser
echo [HINT] keep this window open while using.
echo.
"%PY_CMD%" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
echo.
echo [INFO] server stopped.
pause