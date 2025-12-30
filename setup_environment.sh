#!/bin/bash
################################################################################
# Clinical Valence Testing - Automated Environment Setup
################################################################################
# This script sets up a complete Python virtual environment optimized for
# NVIDIA H100 GPU with all required dependencies.
#
# Usage: bash setup_environment.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
VENV_NAME="venv"
PYTHON_MIN_VERSION="3.8"
CUDA_VERSION="12.1"

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}Clinical Valence Testing - Environment Setup${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Function to compare versions
version_ge() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

# Step 1: Check Python version
echo -e "${YELLOW}[1/9] Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python not found. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "Found Python: $PYTHON_VERSION"

if version_ge "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
    echo -e "${GREEN}[OK] Python version OK${NC}"
else
    echo -e "${RED}Error: Python $PYTHON_MIN_VERSION or higher required, found $PYTHON_VERSION${NC}"
    exit 1
fi

# Step 2: Check for NVIDIA GPU
echo ""
echo -e "${YELLOW}[2/9] Checking for NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Unknown GPU")
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "Unknown")
    CUDA_VERSION_DETECTED=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "Unknown")

    echo -e "${GREEN}[OK] NVIDIA GPU detected${NC}"
    echo -e "  GPU: $GPU_INFO"
    echo -e "  Driver: $DRIVER_VERSION"
    echo -e "  CUDA: $CUDA_VERSION_DETECTED"

    HAS_GPU=true
else
    echo -e "${YELLOW}[WARNING] No NVIDIA GPU detected or nvidia-smi not found${NC}"
    echo -e "${YELLOW}  Installation will continue with CPU-only support${NC}"
    HAS_GPU=false
fi

# Step 3: Check if virtual environment already exists
echo ""
echo -e "${YELLOW}[3/9] Checking for existing virtual environment...${NC}"
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}[WARNING] Virtual environment '$VENV_NAME' already exists${NC}"
    read -p "Do you want to remove it and create a fresh one? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_NAME"
        echo -e "${GREEN}[OK] Removed${NC}"
    else
        echo -e "${YELLOW}Using existing virtual environment${NC}"
    fi
fi

# Step 4: Create virtual environment
if [ ! -d "$VENV_NAME" ]; then
    echo ""
    echo -e "${YELLOW}[4/9] Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv $VENV_NAME
    echo -e "${GREEN}[OK] Virtual environment created: $VENV_NAME${NC}"
else
    echo ""
    echo -e "${YELLOW}[4/9] Virtual environment already exists${NC}"
fi

# Step 5: Activate virtual environment
echo ""
echo -e "${YELLOW}[5/9] Activating virtual environment...${NC}"
source $VENV_NAME/bin/activate
echo -e "${GREEN}[OK] Virtual environment activated${NC}"

# Step 6: Upgrade pip, setuptools, and wheel
echo ""
echo -e "${YELLOW}[6/9] Upgrading pip, setuptools, and wheel...${NC}"
pip install --upgrade pip setuptools wheel --quiet
PIP_VERSION=$(pip --version | awk '{print $2}')
echo -e "${GREEN}[OK] pip $PIP_VERSION${NC}"

# Step 7: Install PyTorch with CUDA support
echo ""
echo -e "${YELLOW}[7/9] Installing PyTorch...${NC}"
if [ "$HAS_GPU" = true ]; then
    echo -e "${CYAN}Installing PyTorch with CUDA $CUDA_VERSION support for GPU acceleration...${NC}"
    echo -e "${CYAN}This may take several minutes...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo -e "${CYAN}Installing PyTorch (CPU-only version)...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo -e "${GREEN}[OK] PyTorch installed${NC}"

# Step 8: Install project dependencies
echo ""
echo -e "${YELLOW}[8/9] Installing project dependencies...${NC}"
echo -e "${CYAN}This may take several minutes...${NC}"
pip install -r requirements.txt --quiet
echo -e "${GREEN}[OK] All dependencies installed${NC}"

# Step 9: Verify installation
echo ""
echo -e "${YELLOW}[9/9] Verifying installation...${NC}"

# Test imports
echo -e "${CYAN}Testing package imports...${NC}"

# Test PyTorch
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo -e "${GREEN}  [PASS] PyTorch $TORCH_VERSION${NC}"
else
    echo -e "${RED}  [FAIL] PyTorch import failed${NC}"
    exit 1
fi

# Test CUDA availability
if [ "$HAS_GPU" = true ]; then
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        echo -e "${GREEN}  [PASS] CUDA available: $GPU_COUNT GPU(s)${NC}"
        echo -e "${GREEN}    GPU: $GPU_NAME${NC}"
    else
        echo -e "${YELLOW}  [WARNING] CUDA not available (PyTorch installed but can't access GPU)${NC}"
    fi
fi

# Test Transformers
if python -c "import transformers; print(f'Transformers {transformers.__version__}')" 2>/dev/null; then
    TRANSFORMERS_VERSION=$(python -c "import transformers; print(transformers.__version__)")
    echo -e "${GREEN}  [PASS] Transformers $TRANSFORMERS_VERSION${NC}"
else
    echo -e "${RED}  [FAIL] Transformers import failed${NC}"
    exit 1
fi

# Test data science packages
if python -c "import numpy, pandas, scipy; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}  [PASS] NumPy, Pandas, SciPy${NC}"
else
    echo -e "${RED}  [FAIL] Data science packages import failed${NC}"
    exit 1
fi

# Test statistical packages
if python -c "import statsmodels, sklearn; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}  [PASS] Statsmodels, Scikit-learn${NC}"
else
    echo -e "${RED}  [FAIL] Statistical packages import failed${NC}"
    exit 1
fi

# Test visualization packages
if python -c "import matplotlib, seaborn, plotly; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}  [PASS] Matplotlib, Seaborn, Plotly${NC}"
else
    echo -e "${RED}  [FAIL] Visualization packages import failed${NC}"
    exit 1
fi

# Test project modules
if python -c "from config_loader import get_config; from logger import setup_logging; from statistical_analysis import StatisticalAnalyzer; print('OK')" 2>/dev/null; then
    echo -e "${GREEN}  [PASS] Project modules${NC}"
else
    echo -e "${RED}  [FAIL] Project modules import failed${NC}"
    exit 1
fi

# Summary
echo ""
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${GREEN}[SUCCESS] Environment Setup Complete${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${CYAN}Environment Details:${NC}"
echo -e "  Python: $PYTHON_VERSION"
echo -e "  Virtual Environment: $VENV_NAME"
if [ "$HAS_GPU" = true ]; then
    echo -e "  GPU Support: Enabled ($GPU_INFO)"
    echo -e "  CUDA: $CUDA_VERSION_DETECTED"
else
    echo -e "  GPU Support: Disabled (CPU only)"
fi
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo -e "  1. Activate the virtual environment:"
echo -e "     ${YELLOW}source $VENV_NAME/bin/activate${NC}"
echo ""
echo -e "  2. Copy your data files to ./data/ directory:"
echo -e "     ${YELLOW}cp /path/to/DIA_GROUPS_3_DIGITS_adm_test.csv ./data/${NC}"
echo -e "     ${YELLOW}cp /path/to/ALL_3_DIGIT_DIA_CODES.txt ./data/${NC}"
echo ""
echo -e "  3. Run the analysis:"
echo -e "     ${YELLOW}bash run_analysis.sh${NC}"
echo ""
echo -e "  4. Or run manually:"
echo -e "     ${YELLOW}python main.py --test_set_path ./data/DIA_GROUPS_3_DIGITS_adm_test.csv --gpu true${NC}"
echo ""
echo -e "${CYAN}Documentation:${NC}"
echo -e "  - Quick Start: ${YELLOW}QUICKSTART.md${NC}"
echo -e "  - Environment: ${YELLOW}ENVIRONMENT_SETUP.md${NC}"
echo -e "  - Full Guide:  ${YELLOW}README.md${NC}"
echo ""
