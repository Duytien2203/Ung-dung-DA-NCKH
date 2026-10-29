@echo off
setlocal
cd /d "%~dp0"

:: Auto-create Streamlit credentials to skip email prompt forever
if not exist "%USERPROFILE%\.streamlit" (
    mkdir "%USERPROFILE%\.streamlit" >nul 2>&1
)
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
    echo [general]> "%USERPROFILE%\.streamlit\credentials.toml"
    echo email = "">> "%USERPROFILE%\.streamlit\credentials.toml"
)

cls
echo ===================================================================
echo     SmartResearch Studio - Nen tang Thong ke Chuan xac NCKH
echo ===================================================================
echo.
echo Dang khoi dong may chu va mo trinh duyet web...
echo (Ban co the thu nho cua so nay, khong can tat cho den khi dung xong)
echo.

"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m streamlit run app.py --browser.gatherUsageStats false

pause
