@echo off
title SmartResearch Studio - Academic Edition

echo =========================================================================
echo    DANG KHOI DONG SMARTRESEARCH STUDIO TAI LOCAL (MAY TINH CUA BAN)...
echo =========================================================================
echo.

cd /d "%~dp0"

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

set PYTHON_BIN=C:\Users\Admin\AppData\Local\Programs\Python\Python312\python.exe

if not exist "%PYTHON_BIN%" (
    set PYTHON_BIN=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
)

if not exist "%PYTHON_BIN%" (
    set PYTHON_BIN=python
)

echo [1/2] Dang mo trinh duyet tai: http://localhost:8502
start "" "http://localhost:8502"

echo [2/2] Dang chay may chu phan tich du lieu...
echo.
echo =========================================================================
echo    UNG DUNG DANG CHAY TAI: http://localhost:8502
echo    (De tat ung dung, ban chi can dong cua so nay lai)
echo =========================================================================
echo.

"%PYTHON_BIN%" -m streamlit run app.py --server.port 8502 --server.headless true

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [THONG BAO] Cong 8502 dang ban, chuyen sang cong 8503...
    start "" "http://localhost:8503"
    "%PYTHON_BIN%" -m streamlit run app.py --server.port 8503 --server.headless true
)

pause
