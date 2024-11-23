@echo off
echo.
echo ================================================================================
echo                          PyPotteryLens Setup
echo ================================================================================
echo.

:: Check CUDA availability through nvidia-smi
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [x] NVIDIA GPU detected
    for /f "tokens=2 delims=," %%a in ('nvidia-smi --query-gpu^=name --format^=csv ^| findstr /v "name"') do set GPU_NAME=%%a
    echo     • GPU: %GPU_NAME%
    
    :: Get CUDA version
    for /f "tokens=3" %%a in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VERSION=%%a
    echo     • CUDA Version: %CUDA_VERSION%
    set CUDA_AVAILABLE=1
) else (
    echo [!] No NVIDIA GPU detected - will install CPU-only version
    set CUDA_AVAILABLE=0
)

echo.
echo Checking Python environment...

:: Check if Python virtual environment exists
set VENV_EXISTS=0
if exist "venv" (
    echo [x] Virtual environment already exists
    set VENV_EXISTS=1
) else (
    echo [*] Creating virtual environment...
    python -m venv venv
    set VENV_EXISTS=0
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Only install packages if venv is newly created
if %VENV_EXISTS%==0 (
    echo [*] New virtual environment detected, installing packages...
    
    :: Update pip first
    python -m pip install --upgrade pip
    
    :: Install PyTorch based on CUDA availability
    if %CUDA_AVAILABLE%==1 (
        echo [*] Installing PyTorch with CUDA support...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo [*] Installing CPU-only PyTorch...
        pip install torch torchvision torchaudio
    )
    
    :: Install other requirements
    echo [*] Installing base packages...
    pip install -r requirements.txt
)

echo [*] Checking model...
python -c "from utils import download_model; exit(0 if download_model() else 1)"
python -c "from utils import download_model; exit(0 if download_model(url='https://huggingface.co/lrncrd/PyPotteryLens/resolve/main/model_classifier.pth', dest_path = 'models_classifier/model_classifier.pth') else 1)"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Error downloading model. Please check your internet connection and try again.
    pause
    exit /b 1
)

:: Delete the temporary download script
:: del download_model.py

:: Verify installation
echo.
echo Verifying PyTorch installation...
python -c "import torch; print(f'[✓] PyTorch {torch.__version__}'); print(f'[✓] CUDA available: {torch.cuda.is_available()}'); print(f'[✓] GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"

:: Start the application
echo.
echo [*] Starting PyPotteryLens...
python app.py

:: Keep the window open if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] An error occurred. Please check the messages above.
    pause
)

pause