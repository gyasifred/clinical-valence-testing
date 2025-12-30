# Changelog

All notable changes to the Clinical Valence Testing project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-01

### Production Release

This is the first stable production release of Clinical Valence Testing with comprehensive features for analyzing valence bias in clinical NLP models.

### Added

#### Core Features
- Complete behavioral testing framework for clinical NLP models
- Four shift types: neutralize, pejorative, laudatory, and neutral valence
- Attention weight extraction and analysis for transformer models
- Comprehensive statistical analysis suite with multiple testing methods
- Interactive visualization dashboards with Plotly

#### Environment & Setup
- Automated environment setup script (`setup_environment.sh`)
- Installation verification script (`verify_installation.py`)
- Automated analysis runner (`run_analysis.sh`)
- Comprehensive environment setup documentation
- Step-by-step quickstart guide for user data files

#### GPU Support
- NVIDIA H100 NVL optimization (95GB memory, batch size 768)
- NVIDIA H100 SXM/PCIe optimization (80GB memory, batch size 512)
- NVIDIA A100 optimization (40-80GB memory, batch size 256)
- Intelligent batch size detection based on GPU memory
- CUDA 12.1+ support with PyTorch 2.1+

#### Data Handling
- Support for DIA_GROUPS_3_DIGITS datasets
- CSV format with configurable column names
- Automatic data validation and format checking
- Data directory structure with README

#### Statistical Analysis
- Paired t-tests and Wilcoxon signed-rank tests
- Effect sizes: Cohen's d and Hedges' g
- Multiple comparison corrections: FDR and Bonferroni
- Bootstrap confidence intervals
- Approximate randomization testing
- Comprehensive statistical reports

#### Documentation
- Complete README with installation and usage instructions
- Quick start guide (QUICKSTART.md)
- Environment setup guide (ENVIRONMENT_SETUP.md)
- Data directory instructions
- Results template with placeholders (results.txt)
- Code documentation with docstrings

#### Configuration
- YAML-based configuration system
- Command-line argument override support
- Reproducibility controls (random seed, deterministic mode)
- Flexible output settings

#### Code Quality
- Professional status markers (no emojis)
- Production-ready scripts
- Comprehensive error handling
- Progress tracking and logging
- Clean, maintainable code structure

### Changed
- Removed all emojis from code for professional production environment
- Updated batch sizes for H100 NVL GPUs (95GB memory)
- Improved status messages with bracket notation ([OK], [PASS], [FAIL], etc.)
- Enhanced documentation organization
- Optimized requirements for H100 compatibility

### Removed
- Redundant documentation files (12 files consolidated)
- Development-specific notes and reviews
- Duplicate README files
- Emojis from all scripts and code

### Fixed
- Package version compatibility issues
- GPU memory detection and batch size optimization
- H100 NVL specific optimizations
- Professional output formatting

### Technical Details

#### Dependencies
- Python 3.8-3.11
- PyTorch 2.1.0+ with CUDA 12.1
- Transformers 4.35.0+
- NumPy <2.0.0 (compatibility)
- Pandas 1.5.0+
- SciPy, Statsmodels, Scikit-learn
- Matplotlib, Seaborn, Plotly
- Fire, TQDM, PyYAML, Pydantic

#### Hardware Requirements
- Recommended: NVIDIA H100 NVL (95GB)
- Minimum: NVIDIA GPU with 16GB+ or CPU
- RAM: 16GB minimum, 32GB+ recommended
- Storage: 10GB+ free space

#### Performance
- H100 NVL: ~3-5 minutes for 3,000-5,000 samples
- H100 SXM/PCIe: ~5-7 minutes for 3,000-5,000 samples
- A100 (80GB): ~8-10 minutes for 3,000-5,000 samples
- CPU: ~30-45 minutes for 3,000-5,000 samples

### Security
- No known vulnerabilities
- Safe for production use
- Input validation implemented
- No external API calls

### Compatibility
- Linux (Ubuntu 20.04/22.04 recommended)
- Windows 10/11
- macOS (CPU only)
- CUDA 12.1, 11.8, or CPU

---

## Release Notes

This version 1.0.0 represents a stable, production-ready release suitable for:
- Academic research and publications
- Clinical NLP model evaluation
- Bias detection in healthcare AI systems
- Production deployment in research environments

All features have been tested and verified for correctness and performance.

---

For detailed installation instructions, see [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md).
For usage instructions, see [QUICKSTART.md](QUICKSTART.md) and [README.md](README.md).
