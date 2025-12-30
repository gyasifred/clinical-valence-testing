#!/usr/bin/env python3
"""
Clinical Valence Testing - Installation Verification Script

This script verifies that all required packages are installed correctly
and that the system is properly configured for running the analysis.
"""

import sys
import platform

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def print_success(text):
    """Print success message"""
    print(f"[PASS] {text}")

def print_error(text):
    """Print error message"""
    print(f"[FAIL] {text}")

def print_warning(text):
    """Print warning message"""
    print(f"[WARNING] {text}")

def check_python_version():
    """Check Python version"""
    print_header("Python Environment")

    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    print(f"Python version: {version_str}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version_str} is supported")
        return True
    else:
        print_error(f"Python {version_str} is not supported. Need Python 3.8+")
        return False

def check_core_packages():
    """Check core deep learning packages"""
    print_header("Core Deep Learning Packages")

    all_ok = True

    # Check PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print_success("PyTorch installed")
    except ImportError:
        print_error("PyTorch not installed")
        all_ok = False
        return all_ok

    # Check CUDA availability
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)

        print(f"CUDA version: {cuda_version}")
        print(f"GPU count: {device_count}")
        print(f"GPU name: {device_name}")
        print_success(f"CUDA available with {device_count} GPU(s)")

        # Check GPU memory
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name}")
            print(f"    Total memory: {memory_gb:.1f} GB")
            print(f"    Compute capability: {props.major}.{props.minor}")
    else:
        print_warning("CUDA not available - will use CPU (slower)")

    # Check Transformers
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
        print_success("Transformers installed")
    except ImportError:
        print_error("Transformers not installed")
        all_ok = False

    return all_ok

def check_data_packages():
    """Check data processing packages"""
    print_header("Data Processing Packages")

    all_ok = True
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
    ]

    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"{display_name}: {version}")
            print_success(f"{display_name} installed")
        except ImportError:
            print_error(f"{display_name} not installed")
            all_ok = False

    return all_ok

def check_statistical_packages():
    """Check statistical analysis packages"""
    print_header("Statistical Analysis Packages")

    all_ok = True
    packages = [
        ('statsmodels', 'Statsmodels'),
        ('sklearn', 'Scikit-learn'),
    ]

    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"{display_name}: {version}")
            print_success(f"{display_name} installed")
        except ImportError:
            print_error(f"{display_name} not installed")
            all_ok = False

    return all_ok

def check_visualization_packages():
    """Check visualization packages"""
    print_header("Visualization Packages")

    all_ok = True
    packages = [
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('plotly', 'Plotly'),
    ]

    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"{display_name}: {version}")
            print_success(f"{display_name} installed")
        except ImportError:
            print_error(f"{display_name} not installed")
            all_ok = False

    return all_ok

def check_utility_packages():
    """Check utility packages"""
    print_header("Utility Packages")

    all_ok = True
    packages = [
        ('fire', 'Fire (CLI)'),
        ('tqdm', 'TQDM (Progress bars)'),
        ('yaml', 'PyYAML (Config)'),
        ('pydantic', 'Pydantic (Validation)'),
    ]

    for module_name, display_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"{display_name}: {version}")
            print_success(f"{display_name} installed")
        except ImportError:
            print_error(f"{display_name} not installed")
            all_ok = False

    return all_ok

def check_project_modules():
    """Check project-specific modules"""
    print_header("Project Modules")

    all_ok = True
    modules = [
        'config_loader',
        'logger',
        'utils',
        'valence_testing',
        'prediction',
        'statistical_analysis',
        'interactive_viz',
    ]

    for module_name in modules:
        try:
            __import__(module_name)
            print_success(f"{module_name}.py")
        except ImportError as e:
            print_error(f"{module_name}.py - {e}")
            all_ok = False

    return all_ok

def check_test_shifts():
    """Check shift modules"""
    print_header("Shift Modules")

    all_ok = True
    shifts = [
        'test_shifts.base_shift',
        'test_shifts.neutralize_shift',
        'test_shifts.pejorative_shift',
        'test_shifts.laudatory_shift',
        'test_shifts.neutralVal_shift',
    ]

    for shift_name in shifts:
        try:
            __import__(shift_name)
            print_success(f"{shift_name}")
        except ImportError as e:
            print_error(f"{shift_name} - {e}")
            all_ok = False

    return all_ok

def run_basic_functionality_tests():
    """Run basic functionality tests"""
    print_header("Basic Functionality Tests")

    all_ok = True

    # Test config loading
    try:
        from config_loader import get_config
        config = get_config()
        print_success("Config loading works")
    except Exception as e:
        print_error(f"Config loading failed: {e}")
        all_ok = False

    # Test logger
    try:
        from logger import setup_logging, get_logger
        setup_logging(level="INFO")
        logger = get_logger("test")
        print_success("Logger works")
    except Exception as e:
        print_error(f"Logger failed: {e}")
        all_ok = False

    # Test PyTorch tensor operations
    try:
        import torch
        x = torch.randn(10, 10)
        y = x @ x.T
        assert y.shape == (10, 10)
        print_success("PyTorch tensor operations work")
    except Exception as e:
        print_error(f"PyTorch operations failed: {e}")
        all_ok = False

    # Test GPU operations if available
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(10, 10).cuda()
            y = x @ x.T
            assert y.is_cuda
            print_success("GPU tensor operations work")
    except Exception as e:
        print_error(f"GPU operations failed: {e}")
        all_ok = False

    return all_ok

def main():
    """Main verification function"""
    print("\n" + "=" * 80)
    print("  CLINICAL VALENCE TESTING - INSTALLATION VERIFICATION")
    print("=" * 80)

    results = []

    results.append(("Python Version", check_python_version()))
    results.append(("Core Packages", check_core_packages()))
    results.append(("Data Packages", check_data_packages()))
    results.append(("Statistical Packages", check_statistical_packages()))
    results.append(("Visualization Packages", check_visualization_packages()))
    results.append(("Utility Packages", check_utility_packages()))
    results.append(("Project Modules", check_project_modules()))
    results.append(("Shift Modules", check_test_shifts()))
    results.append(("Functionality Tests", run_basic_functionality_tests()))

    # Summary
    print_header("Verification Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name:.<40} {status}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n" + "=" * 80)
        print_success("ALL CHECKS PASSED - System is ready!")
        print("=" * 80 + "\n")

        print("You can now run the analysis:")
        print("  bash run_analysis.sh")
        print("or")
        print("  python main.py --test_set_path ./data/DIA_GROUPS_3_DIGITS_adm_test.csv --gpu true")
        print()

        return 0
    else:
        print("\n" + "=" * 80)
        print_error(f"SOME CHECKS FAILED ({total - passed} failures)")
        print("=" * 80 + "\n")

        print("Please fix the issues above before running the analysis.")
        print("See ENVIRONMENT_SETUP.md for detailed installation instructions.")
        print()

        return 1

if __name__ == "__main__":
    sys.exit(main())
