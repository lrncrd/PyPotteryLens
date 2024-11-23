#!/bin/bash

echo "================================================================================"
echo "                          PyPotteryLens Setup"
echo "================================================================================"
echo

# Function to check if NVIDIA GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "[✓] NVIDIA GPU detected"
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
        echo "    • GPU: $GPU_NAME"
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
        echo "    • CUDA Version: $CUDA_VERSION"
        return 0
    else
        echo "[!] No NVIDIA GPU detected - will install CPU-only version"
        return 1
    fi
}

echo "Checking GPU..."
if check_gpu; then
    CUDA_AVAILABLE=1
else
    CUDA_AVAILABLE=0
fi

echo
echo "Checking Python environment..."

# Check if Python virtual environment exists
if [ -d "venv" ]; then
    echo "[✓] Virtual environment already exists"
    VENV_EXISTS=1
else
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
    VENV_EXISTS=0
fi

# Activate virtual environment
source venv/bin/activate

# Only install packages if venv is newly created
if [ $VENV_EXISTS -eq 0 ]; then
    echo "[*] New virtual environment detected, installing packages..."
    
    # Update pip first
    python -m pip install --upgrade pip
    
    # Install PyTorch based on CUDA availability
    if [ $CUDA_AVAILABLE -eq 1 ]; then
        echo "[*] Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        echo "[*] Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio
    fi
    
    # Install other requirements
    echo "[*] Installing base packages..."
    pip install -r requirements.txt
fi

# Download model using utils.py
echo "[*] Checking model..."
if ! python -c "from utils import download_model; exit(0 if download_model() else 1)"; then
    echo
    echo "[!] Error downloading model. Please check your internet connection and try again."
    exit 1
fi

# Download model using utils.py
echo "[*] Checking model..."
if ! python -c "from utils import download_model; exit(0 if download_model(url='https://huggingface.co/lrncrd/PyPotteryLens/resolve/main/model_classifier.pth', dest_path = 'models_classifier/model_classifier.pth') else 1)"; then
    echo
    echo "[!] Error downloading model. Please check your internet connection and try again."
    exit 1
fi

# Verify installation
echo
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'[✓] PyTorch {torch.__version__}'); print(f'[✓] CUDA available: {torch.cuda.is_available()}'); print(f'[✓] GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Start the application
echo
echo "[*] Starting PyPotteryLens..."
python app.py

# Handle errors
if [ $? -ne 0 ]; then
    echo
    echo "[!] An error occurred. Please check the messages above."
    read -p "Press Enter to continue..."
    exit 1
fi

read -p "Press Enter to continue..."