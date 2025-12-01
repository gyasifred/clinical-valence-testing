# Clinical Valence Testing with Attention Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Publication-Ready Research Code** for investigating bias in clinical NLP models through valence testing and attention analysis.

This project extends the work of [Clinical Behavioral Testing](https://github.com/bvanaken/clinical-behavioral-testing) by van Aken et al. to examine how valence words affect clinical predictions and analyze attention patterns in clinical NLP models.

---

## 🆕 What's New (Latest Release)

- ✨ **Configuration Management**: YAML-based configuration system for reproducible experiments
- 📊 **Statistical Analysis**: Comprehensive hypothesis testing, effect sizes, and multiple comparison correction
- 📝 **Professional Logging**: Colored console output, file logging, and experiment tracking
- 🔬 **Publication Ready**: Rigorous statistical methods and reproducibility controls
- 📚 **Enhanced Documentation**: Complete API documentation with examples
- ✅ **Quality Assurance**: Input validation, error handling, and data quality checks

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Statistical Analysis](#statistical-analysis)
  - [Visualization](#visualization)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core Capabilities

- **🔄 Behavioral Testing**: Systematic testing of how valence words affect model predictions
- **🎯 Multiple Shift Types**:
  - Pejorative (negative descriptors: "non-compliant", "difficult")
  - Laudatory (positive descriptors: "compliant", "cooperative")
  - Neutral (objective descriptors: "typical", "presenting")
  - Neutralization (removal of all valence words)
- **👁️ Attention Analysis**: Extraction and analysis of model attention patterns
- **📈 Statistical Analysis**: Rigorous hypothesis testing and effect size calculation
- **📊 Visualization**: Publication-quality figures and heatmaps
- **⚙️ Configurable**: YAML-based configuration for all parameters
- **🔁 Reproducible**: Deterministic mode and seed setting for exact replication

### Advanced Features

- **Checkpointing**: Resume long-running experiments from interruptions
- **Batch Processing**: Efficient processing of large datasets
- **Multiple Comparison Correction**: FDR, Bonferroni methods
- **Effect Size Calculation**: Cohen's d, Hedges' g
- **Bootstrap Confidence Intervals**: Robust uncertainty quantification
- **Quality Checks**: Automatic validation of inputs and outputs
- **Progress Tracking**: Real-time progress reporting with logging

---

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 2.0+
- CUDA-capable GPU (recommended but not required)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gyasifred/clinical-valence-testing.git
   cd clinical-valence-testing
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```python
   python -c "import transformers, torch; print('Installation successful!')"
   ```

---

## Quick Start

### 1. Configure Your Experiment

Edit `config.yaml` or create a custom configuration:

```yaml
model:
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"
  batch_size: 128
  attention:
    layer_num: 11
    head_num: 11

analysis:
  significance_level: 0.05
  multiple_testing_correction: "fdr_bh"

random_seed: 42
```

### 2. Run Behavioral Tests

```bash
python main.py \
  --test_set_path data/test.csv \
  --model_path models/checkpoint \
  --shift_keys neutralize,pejorative,laud,neutralval \
  --task diagnosis \
  --save_dir results/
```

### 3. Analyze Results

```python
from statistical_analysis import StatisticalAnalyzer
import pandas as pd

# Load results
baseline = pd.read_csv('results/neutralize_diagnosis.csv')
treatment = pd.read_csv('results/pejorative_diagnosis.csv')

# Analyze
analyzer = StatisticalAnalyzer(
    significance_level=0.05,
    correction_method='fdr_bh'
)

results = analyzer.analyze_diagnosis_shifts(
    baseline_probs=baseline,
    treatment_probs=treatment,
    diagnosis_codes=code_list
)

# Generate report
report = analyzer.generate_analysis_report(
    diagnosis_results=results,
    output_path='statistical_report.txt'
)
```

---

## Configuration

### Configuration System

The project uses a YAML-based configuration system for all parameters:

```python
from config_loader import get_config

config = get_config()  # Loads config.yaml by default

# Access configuration
print(config.model.batch_size)
print(config.random_seed)
print(config.analysis.significance_level)
```

### Key Configuration Sections

#### Model Configuration
```yaml
model:
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"
  max_length: 512
  batch_size: 128
  device: "auto"  # auto, cuda, or cpu
```

#### Analysis Configuration
```yaml
analysis:
  baseline: "neutralize"
  significance_level: 0.05
  multiple_testing_correction: "fdr_bh"
  effect_size_threshold: 0.01
  statistical_tests:
    - "paired_ttest"
    - "wilcoxon"
    - "effect_size"
```

#### Reproducibility Configuration
```yaml
random_seed: 42
deterministic: true
```

See `config.yaml` for complete configuration options.

---

## Usage

### Basic Usage

#### Running Behavioral Tests

```python
from valence_testing import BehavioralTesting
from prediction import DiagnosisPredictor
from test_shifts.pejorative_shift import PejorativeShift
from config_loader import get_config
from logger import setup_logging

# Setup
config = get_config()
setup_logging(level="INFO", log_file="experiment.log")

# Initialize predictor
predictor = DiagnosisPredictor(
    checkpoint_path="models/checkpoint",
    test_set_path="data/test.csv",
    batch_size=config.model.batch_size
)

# Initialize testing framework
bt = BehavioralTesting(test_dataset_path="data/test.csv")

# Create shift
shift = PejorativeShift()

# Run test
stats = bt.run_test(shift, predictor, "results/pejorative.csv")
```

### Statistical Analysis

#### Comprehensive Analysis

```python
from statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer(
    significance_level=0.05,
    correction_method="fdr_bh"
)

# Paired t-test
result = analyzer.paired_ttest(baseline, treatment)
print(f"p-value: {result.p_value}")
print(f"Effect size (Cohen's d): {result.effect_size}")
print(f"95% CI: {result.confidence_interval}")

# Wilcoxon test (non-parametric)
result = analyzer.wilcoxon_test(baseline, treatment)

# Bootstrap confidence interval
estimate, ci = analyzer.bootstrap_confidence_interval(
    data=differences,
    statistic_func=np.mean,
    n_bootstrap=10000
)
```

#### Diagnosis Shift Analysis

```python
results_df = analyzer.analyze_diagnosis_shifts(
    baseline_probs=baseline_df,
    treatment_probs=treatment_df,
    diagnosis_codes=['401', '250', '276', ...]
)

# Results include:
# - mean_shift, median_shift
# - Cohen's d, Hedges' g
# - t-test and Wilcoxon p-values
# - Corrected p-values
# - Confidence intervals
```

### Visualization

```python
from analysis_files.plot import ValenceShiftAnalyzer

analyzer = ValenceShiftAnalyzer(
    results_dir="results/",
    config=visualization_config
)

# Plot diagnosis probability shifts
analyzer.plot_valence_shifts(save_path="figures/valence_shifts.png")

# Plot attention weight shifts
analyzer.plot_attention_shifts(save_path="figures/attention_shifts.png")
```

### Reproducibility

```python
from utils import set_random_seeds, configure_deterministic_mode

# Set seeds for reproducibility
set_random_seeds(seed=42)

# Enable deterministic mode (slower but reproducible)
configure_deterministic_mode(deterministic=True)
```

---

## Project Structure

```
clinical-valence-testing/
├── config.yaml                 # Configuration file
├── config_loader.py            # Configuration management
├── logger.py                   # Logging framework
├── statistical_analysis.py     # Statistical analysis module
├── utils.py                    # Utility functions
├── requirements.txt            # Dependencies
│
├── main.py                     # CLI entry point
├── prediction.py               # Prediction classes
├── valence_testing.py          # Testing framework
│
├── test_shifts/                # Shift implementations
│   ├── base_shift.py
│   ├── pejorative_shift.py
│   ├── laudatory_shift.py
│   ├── neutralVal_shift.py
│   └── neutralize_shift.py
│
├── analysis_files/             # Analysis and visualization
│   ├── plot.py
│   ├── plot_word_attention_shift.py
│   └── analysis_result/
│
├── tests/                      # Unit tests
│   ├── test_config.py
│   ├── test_statistics.py
│   └── test_utils.py
│
├── examples/                   # Example notebooks
│   ├── 01_getting_started.ipynb
│   └── 02_statistical_analysis.ipynb
│
└── docs/                       # Documentation
    ├── api/
    └── user_guide/
```

---

## Documentation

### Available Documentation

- **API Documentation**: Complete reference for all modules and functions
- **User Guide**: Step-by-step tutorials and best practices
- **Statistical Methods**: Explanation of all statistical tests used
- **Configuration Guide**: Detailed configuration options
- **Examples**: Jupyter notebooks with complete workflows

### Building Documentation

```bash
cd docs
make html
```

Documentation will be available in `docs/_build/html/`.

---

## Citation

If you use this code in your research, please cite both this project and the original Clinical Behavioral Testing work:

### Original Work

```bibtex
@inproceedings{vanAken2021,
   author    = {Betty van Aken and
                Sebastian Herrmann and
                Alexander Löser},
   title     = {What Do You See in this Patient? Behavioral Testing of Clinical NLP Models},
   booktitle = {Bridging the Gap: From Machine Learning Research to Clinical Practice,
                Research2Clinics Workshop @ NeurIPS 2021},
   year      = {2021}
}
```

### This Project

```bibtex
@software{clinical_valence_testing,
   author    = {Fred Gyasi},
   title     = {Clinical Valence Testing with Attention Analysis},
   year      = {2025},
   url       = {https://github.com/gyasifred/clinical-valence-testing}
}
```

---

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Code Style**: Use Black for formatting, follow PEP 8
2. **Documentation**: Add docstrings to all functions
3. **Testing**: Write unit tests for new features
4. **Logging**: Use the logger, not print statements
5. **Configuration**: Add new parameters to `config.yaml`

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run code formatting
black .
isort .

# Type checking
mypy .
```

---

## FAQ

### Q: How do I add a new shift type?

A: Create a new class inheriting from `BaseShift` in `test_shifts/`:

```python
from test_shifts.base_shift import BaseShift
from enum import Enum

class MyShiftLevel(Enum):
    LEVEL1 = 1
    LEVEL2 = 2

class MyShift(BaseShift):
    def get_groups(self):
        return list(MyShiftLevel)

    def get_group_names(self):
        return [group.name for group in self.get_groups()]

    def get_shift_method(self, sample: str, group: MyShiftLevel):
        # Implement your shift logic
        return modified_sample
```

### Q: How do I ensure reproducibility?

A: Use the configuration system and reproducibility utilities:

```python
from config_loader import get_config
from utils import set_random_seeds, configure_deterministic_mode

config = get_config()
set_random_seeds(config.random_seed)
configure_deterministic_mode(config.deterministic)
```

### Q: What statistical tests should I use?

A: Use paired t-test for normal distributions, Wilcoxon for non-normal:

```python
# Check normality first
from scipy.stats import shapiro
stat, p = shapiro(differences)

if p > 0.05:
    # Use t-test
    result = analyzer.paired_ttest(baseline, treatment)
else:
    # Use Wilcoxon
    result = analyzer.wilcoxon_test(baseline, treatment)
```

### Q: How do I handle large datasets?

A: Use batch processing and checkpointing:

```python
predictor = DiagnosisPredictor(
    checkpoint_path="model.pt",
    test_set_path="large_dataset.csv",
    batch_size=64,  # Adjust based on memory
    checkpoint_interval=1000  # Save every 1000 samples
)
```

---

## Known Issues

- Deterministic mode may not work with all PyTorch operations on GPU
- Very long texts (>512 tokens) are truncated
- Memory usage scales with batch size

See [Issues](https://github.com/gyasifred/clinical-valence-testing/issues) for more details.

---

## Roadmap

- [ ] Interactive web dashboard for results exploration
- [ ] Support for additional NLP models (GPT, RoBERTa)
- [ ] Multi-language support
- [ ] Real-time prediction API
- [ ] Integration with clinical databases

---

## License

This project follows the licensing terms of the original Clinical Behavioral Testing project.

---

## Acknowledgments

- Betty van Aken, Sebastian Herrmann, and Alexander Löser for the original Clinical Behavioral Testing framework
- Hugging Face for the Transformers library
- The BioBERT team for pre-trained clinical models

---

## Contact

- **Author**: Fred Gyasi
- **Repository**: [https://github.com/gyasifred/clinical-valence-testing](https://github.com/gyasifred/clinical-valence-testing)
- **Issues**: [https://github.com/gyasifred/clinical-valence-testing/issues](https://github.com/gyasifred/clinical-valence-testing/issues)

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
