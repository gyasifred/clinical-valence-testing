# Virtual Environment Setup for Clinical Valence Testing

Complete guide for setting up a Python virtual environment with H100 GPU support for the Clinical Valence Testing framework.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Setup (Recommended)](#quick-setup-recommended)
3. [Manual Setup](#manual-setup)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [GPU Configuration](#gpu-configuration)

---

## System Requirements

### Hardware Requirements

- **CPU**: Multi-core processor (Intel/AMD)
- **RAM**: Minimum 16GB, recommended 32GB or more
- **GPU**: NVIDIA H100 (or other CUDA-capable GPU)
- **Storage**: At least 10GB free space for dependencies and model cache

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04/22.04 recommended), Windows 10/11, or macOS
- **Python**: Version 3.8, 3.9, 3.10, or 3.11
- **CUDA**: Version 12.1 or higher (for H100 GPU)
- **NVIDIA Driver**: 525.60.13 or higher (for CUDA 12.1)

### Check Your System

```bash
# Check Python version (should be 3.8-3.11)
python --version
# or
python3 --version

# Check NVIDIA GPU and driver
nvidia-smi

# Check CUDA version
nvcc --version
# or check from nvidia-smi output
```

---

## Quick Setup (Recommended)

### Option 1: Automated Setup Script

We provide an automated setup script that handles everything:

```bash
# 1. Navigate to project directory
cd clinical-valence-testing

# 2. Run the automated setup script
bash setup_environment.sh
```

The script will:
- Check Python version
- Check GPU availability
- Create virtual environment
- Install all dependencies with H100 optimization
- Verify installation
- Run test imports

---

## Manual Setup

### Step 1: Install Python (if needed)

#### On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

#### On CentOS/RHEL:
```bash
sudo yum install python3.10 python3.10-devel
```

#### On Windows:
Download Python from [python.org](https://www.python.org/downloads/) and install.

#### On macOS:
```bash
brew install python@3.10
```

### Step 2: Create Virtual Environment

```bash
# Navigate to project directory
cd clinical-valence-testing

# Create virtual environment
python3 -m venv venv

# Alternative: specify Python version explicitly
python3.10 -m venv venv
```

### Step 3: Activate Virtual Environment

#### On Linux/macOS:
```bash
source venv/bin/activate
```

#### On Windows (Command Prompt):
```cmd
venv\Scripts\activate.bat
```

#### On Windows (PowerShell):
```powershell
venv\Scripts\Activate.ps1
```

**Note**: If you get a permission error on Windows PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Upgrade pip, setuptools, and wheel

```bash
pip install --upgrade pip setuptools wheel
```

### Step 5: Install PyTorch with CUDA Support for H100

**For H100 GPU with CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For other CUDA versions:**
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU only (not recommended for large datasets)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 6: Install Project Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 7: Install Optional Packages (if needed)

```bash
# For development/testing
pip install pytest pytest-cov black flake8 mypy isort

# For Jupyter notebooks
pip install jupyter ipykernel ipywidgets notebook

# For multi-GPU support
pip install accelerate

# For documentation building
pip install sphinx sphinx-rtd-theme
```

---

## Verification

### Test Installation

Run the verification script:

```bash
python verify_installation.py
```

Or manually verify:

```bash
# Test Python imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import pandas, numpy, scipy; print('[OK] Data science packages')"

# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Test GPU detection
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"

# Test all project imports
python -c "from config_loader import get_config; from logger import setup_logging; from statistical_analysis import StatisticalAnalyzer; print('[OK] All project imports successful')"
```

### Expected Output

For successful installation, you should see:

```
PyTorch: 2.1.0+cu121
Transformers: 4.35.0
[OK] Data science packages OK
CUDA available: True
CUDA version: 12.1
GPU count: 1
GPU name: NVIDIA H100 PCIe
[OK] All project imports successful
```

### Check Package Versions

```bash
# List all installed packages
pip list

# Check specific package versions
pip show torch transformers numpy pandas scipy
```

---

## GPU Configuration

### Configure for H100 GPU

The `config.yaml` is pre-configured for H100 NVL:

```yaml
model:
  use_gpu: true
  device: "cuda"
  batch_size: 768  # Optimized for H100 NVL (95GB memory)
  attention:
    aggregation: "average"  # Prevents sub-token bias in attention analysis
```

### Optimize Batch Size for H100

The H100 has different memory configurations:

| H100 Model | Memory | Recommended Batch Size |
|------------|--------|------------------------|
| H100 SXM5  | 80GB   | 512                    |
| H100 PCIe  | 80GB   | 512                    |
| H100 NVL   | 95GB   | 768                    |

The project is pre-configured for H100 NVL (batch_size=768 in config.yaml). Start with this default and adjust if needed:

```bash
# Use default H100 NVL configuration
python main.py --gpu true

# Or manually specify for H100 SXM/PCIe
python main.py --batch_size 512 --gpu true

# Reduce if encountering OOM errors
python main.py --batch_size 256 --gpu true
```

### Monitor GPU Usage

```bash
# Monitor GPU in real-time (run in separate terminal)
watch -n 1 nvidia-smi

# Or use
nvidia-smi dmon -s mu
```

### Enable Mixed Precision (Optional)

For faster training with H100 (supports FP8):

```python
# In your script
import torch
torch.set_float32_matmul_precision('high')  # or 'medium'
```

---

## Troubleshooting

### Issue 1: CUDA Not Available

**Symptoms:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions:**

1. **Check NVIDIA driver:**
   ```bash
   nvidia-smi
   # Should show driver version and GPU
   ```

2. **Reinstall PyTorch with correct CUDA version:**
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Check CUDA installation:**
   ```bash
   nvcc --version
   ```

### Issue 2: Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python main.py --batch_size 128
   ```

2. **Clear GPU cache:**
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Monitor GPU memory:**
   ```bash
   nvidia-smi
   ```

### Issue 3: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solutions:**

1. **Ensure virtual environment is activated:**
   ```bash
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

### Issue 4: Version Conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solutions:**

1. **Create fresh virtual environment:**
   ```bash
   deactivate
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Install in order:**
   ```bash
   # Install PyTorch first
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

   # Then other packages
   pip install -r requirements.txt
   ```

### Issue 5: Slow Installation

**Solutions:**

1. **Use pip cache:**
   ```bash
   pip install -r requirements.txt --cache-dir /tmp/pip-cache
   ```

2. **Increase timeout:**
   ```bash
   pip install -r requirements.txt --default-timeout=100
   ```

3. **Use mirror (China users):**
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

### Issue 6: Permission Denied

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**

1. **Don't use sudo with virtual environment**
2. **Check directory permissions:**
   ```bash
   ls -la venv/
   chmod -R u+w venv/
   ```

---

## Environment Management

### Deactivate Virtual Environment

```bash
deactivate
```

### Remove Virtual Environment

```bash
# Make sure you're deactivated first
deactivate

# Remove the directory
rm -rf venv/
```

### Export Environment

```bash
# Export exact versions
pip freeze > requirements-frozen.txt

# Share with others
git add requirements-frozen.txt
git commit -m "Add frozen requirements"
```

### Recreate Environment from Export

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-frozen.txt
```

---

## Best Practices

1. **Always activate virtual environment** before running scripts
2. **Keep requirements.txt updated** when adding new packages
3. **Use separate environments** for different projects
4. **Don't commit venv/** to git (already in .gitignore)
5. **Document any system-level dependencies** (like CUDA)
6. **Test installation** after setting up on new machine
7. **Pin package versions** for reproducibility

---

## Quick Reference

### Common Commands

```bash
# Create environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_installation.py

# Deactivate
deactivate

# Update package
pip install --upgrade package_name

# List installed packages
pip list

# Show package info
pip show package_name
```

---

## Additional Resources

- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA documentation](https://docs.nvidia.com/cuda/)
- [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)

---

## Support

If you encounter issues not covered here:

1. Check [GitHub Issues](https://github.com/gyasifred/clinical-valence-testing/issues)
2. Review the main [README.md](README.md)
3. Check NVIDIA driver compatibility
4. Verify Python version compatibility
5. Open a new issue with detailed error messages
