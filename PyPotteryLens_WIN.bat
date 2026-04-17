@echo off
echo ====================================
echo  PyPotteryLens Application
echo ====================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [*] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [!] Error: Failed to create virtual environment
        echo [!] Make sure Python is installed and in PATH
        pause
        exit /b 1
    )
    echo [✓] Virtual environment created
) else (
    echo [✓] Virtual environment found
)

echo.
echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

if errorlevel 1 (
    echo [!] Error: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo [*] Checking dependencies...
python -c "import flask" 2>nul
if errorlevel 1 (
    echo [*] Installing dependencies from requirements.txt...
    echo     This may take a few minutes...
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [!] Error: Failed to install dependencies
        pause
        exit /b 1
    )
    echo [✓] Dependencies installed successfully
) else (
    echo [✓] Dependencies already installed
)

echo.
echo ====================================
echo  Starting Flask Server
echo ====================================
echo.
echo Open your browser at: http://localhost:5001
echo.
echo Press Ctrl+C to stop the server
echo.
echo ====================================
echo.

python app.py

deactivate
pause
