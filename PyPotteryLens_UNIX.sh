#!/bin/bash

echo "===================================="
echo " PyPotteryLens Application"
echo "===================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[!] Error: Failed to create virtual environment"
        echo "[!] Make sure Python 3 is installed"
        exit 1
    fi
    echo "[✓] Virtual environment created"
else
    echo "[✓] Virtual environment found"
fi

echo ""
echo "[*] Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "[!] Error: Failed to activate virtual environment"
    exit 1
fi

echo ""
echo "[*] Checking dependencies..."
python -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[*] Installing dependencies from requirements.txt..."
    echo "    This may take a few minutes..."
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[!] Error: Failed to install dependencies"
        exit 1
    fi
    echo "[✓] Dependencies installed successfully"
else
    echo "[✓] Dependencies already installed"
fi

echo ""
echo "===================================="
echo " Starting Flask Server"
echo "===================================="
echo ""
echo "Open your browser at: http://localhost:5001"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "===================================="
echo ""

python app.py

deactivate
