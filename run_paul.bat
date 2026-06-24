@echo off
title Paul the AI Predictor - Server
color 0B

echo ===================================================
echo 🐙 Starting Paul the AI Predictor...
echo ===================================================
echo.

:: Navigate to the project directory
cd /d "D:\FIFA_Predictios_App"

:: Check if the virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\activate.bat
    echo Please make sure the path is correct.
    pause
    exit /b
)

echo [1/3] Activating Virtual Environment...
call .venv\Scripts\activate.bat

echo [2/3] Opening Frontend UI in your browser...
:: 'start' opens the file in the default application (your web browser)
start index.html

echo [3/3] Booting up the Python API Backend...
echo.
:: Start the uvicorn server
uvicorn Paul:app --reload

:: If uvicorn crashes or is closed, pause so you can read any error messages
pause