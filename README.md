# Clinical Valence Testing with Attention Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive framework for testing how valence-laden language (pejorative, laudatory, neutral) affects clinical NLP model predictions. This project extends the work of [Clinical Behavioral Testing](https://github.com/bvanaken/clinical-behavioral-testing) by van Aken et al. with attention analysis and statistical rigor for publication-ready research.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
- [Configuration](#configuration)
- [Understanding the Output](#understanding-the-output)
- [Project Structure](#project-structure)
- [Advanced Usage](#advanced-usage)
- [Citation](#citation)

## ✨ Features

### Core Capabilities
- **🔬 Behavioral Testing**: Systematic testing of clinical NLP models with linguistic perturbations
- **🧠 Attention Analysis**: Extract and analyze transformer attention patterns
- **📊 Statistical Analysis**: Comprehensive hypothesis testing, effect sizes, and multiple comparison corrections
- **🎨 Interactive Visualization**: Plotly-based dashboards for exploring results
- **⚙️ Configuration Management**: YAML-based config with command-line override
- **📝 Comprehensive Logging**: Colored console output and file logging with progress tracking
- **🔁 Reproducibility**: Random seed control and deterministic mode for exact replication

### Shift Types
1. **Pejorative Shift**: Tests impact of negative patient descriptors (e.g., "difficult", "non-compliant")
2. **Laudatory Shift**: Tests impact of positive patient descriptors (e.g., "cooperative", "pleasant")
3. **Neutral Valence Shift**: Tests objective descriptors (e.g., "alert", "oriented")
4. **Neutralize Shift**: Removes all valence terms to create baseline

### Statistical Analysis
- Paired t-tests and Wilcoxon signed-rank tests
- Effect sizes (Cohen's d, Hedges' g)
- Multiple comparison correction (FDR, Bonferroni)
- Bootstrap confidence intervals
- Comprehensive analysis reports

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/gyasifred/clinical-valence-testing.git
cd clinical-valence-testing
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; import transformers; print('✓ Installation successful!')"
```

## ⚡ Quick Start

### Minimal Example

Run all shifts with default configuration:

```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1
```

This will:
1. Load configuration from `config.yaml`
2. Run all four shift types (neutralize, pejorative, laud, neutralval)
3. Save results to `./results/` directory
4. Generate statistical analysis and visualizations

## 📖 Step-by-Step Usage Guide

### Step 1: Prepare Your Data

Your test dataset should be a CSV file with at least two columns:
- **Clinical text column**: Contains clinical notes/texts
- **Code column**: Contains diagnosis codes (ICD codes)

**Example CSV format:**
```csv
text,short_codes
"Patient presents with chest pain and shortness of breath. Alert and oriented.","['I21.9','R07.9']"
"The cooperative patient reports abdominal pain. Vitals stable.","['R10.9']"
```

**Data location:** Place your test data file in an accessible directory (e.g., `./data/test_set.csv`)

### Step 2: Configure the System

Edit `config.yaml` to match your setup:

```yaml
# Model Configuration
model:
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"  # Or your model path
  batch_size: 128
  use_gpu: true
  attention:
    layer_num: 11
    head_num: 11

# Data Configuration
data:
  test_set_path: "./data/test_set.csv"
  text_column: "text"
  code_column: "short_codes"

# Output Configuration
output:
  results_dir: "./results"
  save_attention: true
  save_predictions: true

# Reproducibility
random_seed: 42
deterministic: true
```

### Step 3: Run Basic Behavioral Tests

**Command:**
```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --save_dir ./results
```

**What happens:**
1. ✅ Configuration loaded from `config.yaml`
2. ✅ Random seed set to 42 (reproducible results)
3. ✅ All four shifts initialized (neutralize, pejorative, laud, neutralval)
4. ✅ Model loaded: `bvanaken/CORe-clinical-outcome-biobert-v1`
5. ✅ Behavioral testing runs for each shift:
   - Original samples loaded
   - Text perturbations applied
   - Model predictions generated
   - Attention weights extracted
   - Results compared across shifts
6. ✅ Statistical analysis performed
7. ✅ Results saved to `./results/`

**Expected output files:**
```
./results/
├── neutralize_shift_diagnosis.csv        # Predictions for neutralize shift
├── neutralize_shift_diagnosis_stats.txt  # Statistics for neutralize shift
├── pejorative_shift_diagnosis.csv        # Predictions for pejorative shift
├── pejorative_shift_diagnosis_stats.txt  # Statistics for pejorative shift
├── laud_shift_diagnosis.csv              # Predictions for laudatory shift
├── laud_shift_diagnosis_stats.txt        # Statistics for laudatory shift
├── neutralval_shift_diagnosis.csv        # Predictions for neutral valence shift
├── neutralval_shift_diagnosis_stats.txt  # Statistics for neutral valence shift
└── statistical_analysis.txt              # Comprehensive statistical report
```

### Step 4: Run Specific Shifts Only

To run only specific shifts:

```bash
# Test only pejorative language impact
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --shift_keys pejorative \
  --save_dir ./results/pejorative_only
```

```bash
# Test pejorative and laudatory shifts
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --shift_keys pejorative,laud \
  --save_dir ./results/valence_comparison
```

### Step 5: Adjust Performance Parameters

**For GPU acceleration:**
```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --gpu true \
  --batch_size 256
```

**For CPU-only systems (slower):**
```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --gpu false \
  --batch_size 32
```

### Step 6: Customize Attention Analysis

Analyze different attention heads and layers:

```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --layer_num 9 \
  --head_num 5
```

**Understanding attention parameters:**
- `layer_num`: Which transformer layer to analyze (0-11 for BioBERT)
- `head_num`: Which attention head to analyze (0-11 for BioBERT)
- Higher layers (9-11) typically capture more semantic information
- Different heads may focus on different linguistic patterns

### Step 7: Generate Interactive Visualizations

After running tests, create interactive visualizations:

```python
from interactive_viz import InteractiveVisualizer
import pandas as pd

# Load your results
results = {
    'pejorative': pd.read_csv('./results/pejorative_shift_diagnosis.csv'),
    'laud': pd.read_csv('./results/laud_shift_diagnosis.csv'),
    'neutralval': pd.read_csv('./results/neutralval_shift_diagnosis.csv')
}

# Create visualizer
viz = InteractiveVisualizer()

# Generate comprehensive dashboard
viz.create_dashboard(results, output_path='./results/dashboard.html')

# Open dashboard in browser
import webbrowser
webbrowser.open('./results/dashboard.html')
```

**Dashboard includes:**
- 📊 Box plots comparing diagnosis probabilities across valence types
- 🎻 Violin plots showing attention weight distributions
- 🔥 Heatmaps of diagnosis probability shifts
- 📈 3D surface plots of prediction changes

### Step 8: Run Statistical Analysis

The statistical analysis runs automatically, but you can customize it:

```bash
# Run without statistical analysis (faster)
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --run_statistical_analysis false
```

**What's included in statistical analysis:**
1. **Paired t-tests**: Compare predictions before/after shifts
2. **Effect sizes**: Cohen's d and Hedges' g
3. **Non-parametric tests**: Wilcoxon signed-rank for non-normal distributions
4. **Multiple comparison correction**: FDR and Bonferroni methods
5. **Bootstrap confidence intervals**: 95% CI for robust estimates

### Step 9: Set Custom Random Seeds

For exact reproducibility:

```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --random_seed 123
```

**Reproducibility guarantees:**
- ✅ Same sample selection
- ✅ Same word insertions/removals
- ✅ Same model predictions (with deterministic mode)
- ✅ Same statistical results

### Step 10: Use Custom Configuration Files

```bash
python main.py \
  --config_path ./configs/my_experiment.yaml \
  --test_set_path ./data/test_set.csv
```

Create custom config at `./configs/my_experiment.yaml`:
```yaml
model:
  name: "path/to/your/model"
  batch_size: 64

random_seed: 999
deterministic: true

output:
  results_dir: "./results/experiment_1"
```

## ⚙️ Configuration

### Configuration Hierarchy

Parameters can be set in three ways (in order of precedence):
1. **Command-line arguments** (highest priority)
2. **Configuration file** (`config.yaml`)
3. **Default values** (lowest priority)

### Complete Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_set_path` | str | from config | Path to test dataset CSV |
| `model_path` | str | from config | Path to model checkpoint or HuggingFace model ID |
| `shift_keys` | str | all shifts | Comma-separated shift names |
| `task` | str | "diagnosis" | Prediction task type |
| `save_dir` | str | "./results" | Output directory |
| `gpu` | bool | false | Use GPU for inference |
| `batch_size` | int | 128 | Batch size for predictions |
| `head_num` | int | 11 | Attention head to analyze |
| `layer_num` | int | 11 | Model layer to analyze |
| `code_label` | str | "short_codes" | Column name for diagnosis codes |
| `checkpoint_interval` | int | 1000 | Save checkpoint every N samples |
| `config_path` | str | "./config.yaml" | Path to configuration file |
| `random_seed` | int | 42 | Random seed for reproducibility |
| `run_statistical_analysis` | bool | true | Run statistical analysis on results |

### Available Shifts

| Shift Key | Description | Purpose |
|-----------|-------------|---------|
| `neutralize` | Removes all valence terms | Baseline for comparison |
| `pejorative` | Tests negative descriptors | Understand bias against stigmatized patients |
| `laud` | Tests positive descriptors | Understand favorable bias |
| `neutralval` | Tests neutral descriptors | Understand impact of objective language |

## 📊 Understanding the Output

### 1. CSV Result Files

Each shift produces a CSV with predictions:

```csv
sample_id,original_text,shifted_text,original_prediction,shifted_prediction,prediction_diff,attention_weights
0,"Patient is alert...","Patient is cooperative...","['I10','E11.9']","['I10','E11.9','Z71.3']",0.15,"[0.12,0.08,...]"
```

**Columns:**
- `sample_id`: Unique identifier for each sample
- `original_text`: Original clinical text
- `shifted_text`: Text after applying shift
- `original_prediction`: Model's diagnosis predictions for original text
- `shifted_prediction`: Model's diagnosis predictions for shifted text
- `prediction_diff`: Magnitude of change in predictions
- `attention_weights`: Extracted attention weights (if enabled)

### 2. Statistics Files

Plain text summary of each shift's impact:

```
Shift Statistics: Pejorative
============================
Total samples: 1000
Samples with shift applied: 847
Samples skipped: 153
Average prediction change: 0.085
Max prediction change: 0.42
Min prediction change: 0.001
```

### 3. Statistical Analysis Report

Comprehensive statistical analysis (`statistical_analysis.txt`):

```
COMPREHENSIVE STATISTICAL ANALYSIS
===================================

1. PAIRED T-TESTS
-----------------
Pejorative vs Neutralize:
  t-statistic: 5.23
  p-value: 0.0001 ***
  Cohen's d: 0.34 (medium effect)
  95% CI: [0.052, 0.118]

2. EFFECT SIZES
---------------
Pejorative: Cohen's d = 0.34 (medium)
Laudatory: Cohen's d = -0.21 (small)

3. MULTIPLE COMPARISON CORRECTION
----------------------------------
FDR-corrected p-values:
  Pejorative: 0.0002 ***
  Laudatory: 0.023 *
```

## 📁 Project Structure

```
clinical-valence-testing/
├── main.py                    # Entry point - run experiments
├── config.yaml               # Configuration file
├── config_loader.py          # Configuration management
├── logger.py                 # Logging framework
├── statistical_analysis.py   # Statistical analysis suite
├── interactive_viz.py        # Visualization tools
├── prediction.py             # Model prediction classes
├── utils.py                  # Utility functions
├── valence_testing.py        # Core behavioral testing
├── requirements.txt          # Python dependencies
├── test_shifts/              # Shift implementations
│   ├── base_shift.py         # Base shift class
│   ├── pejorative_shift.py   # Pejorative language shift
│   ├── laudatory_shift.py    # Laudatory language shift
│   ├── neutralVal_shift.py   # Neutral valence shift
│   └── neutralize_shift.py   # Neutralization shift
├── data/                     # Data directory (user-provided)
│   └── test_set.csv          # Test dataset
├── results/                  # Output directory (auto-created)
└── docs/                     # Documentation
    ├── USAGE_GUIDE.md        # Detailed usage guide
    ├── CONFIGURATION_GUIDE.md # Configuration reference
    └── API_REFERENCE.md      # API documentation
```

## 🔬 Advanced Usage

### Custom Shift Implementation

Create your own shift by extending `BaseShift`:

```python
from test_shifts.base_shift import BaseShift
from enum import Enum

class MyCustomLevel(Enum):
    LEVEL_1 = 1
    LEVEL_2 = 2

class MyCustomShift(BaseShift):
    def __init__(self, random_seed=None):
        self.random_seed = random_seed
        # Initialize your shift

    def get_groups(self):
        return list(MyCustomLevel)

    def get_shift_method(self, sample: str, group):
        # Implement your transformation
        return transformed_sample

    def identify_group_in_text(self, text: str):
        # Identify which group this text belongs to
        return MyCustomLevel.LEVEL_1 or None
```

Register your shift in `main.py`:

```python
from test_shifts.my_custom_shift import MyCustomShift

def get_shift_map(random_seed=None):
    return {
        # ... existing shifts ...
        "mycustom": MyCustomShift(random_seed=random_seed)
    }
```

### Batch Processing Multiple Datasets

```bash
#!/bin/bash
for dataset in dataset1.csv dataset2.csv dataset3.csv; do
    python main.py \
      --test_set_path ./data/$dataset \
      --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
      --save_dir ./results/${dataset%.csv}
done
```

### Using Different Models

```bash
# Use a local model
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path /path/to/local/model/checkpoint

# Use different HuggingFace models
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path emilyalsentzer/Bio_ClinicalBERT
```

### Debug Mode

Enable verbose logging for debugging:

Edit `config.yaml`:
```yaml
logging:
  level: "DEBUG"
  console_output: true
  file_output: true
  log_file: "./logs/debug.log"
```

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Out of Memory (OOM) Error**
```bash
# Reduce batch size
python main.py --batch_size 32

# Or disable GPU
python main.py --gpu false
```

**Issue 2: Model Download Fails**
```bash
# Use local model path
python main.py --model_path /local/path/to/model
```

**Issue 3: Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Issue 4: Inconsistent Results**
```bash
# Enable deterministic mode
python main.py --random_seed 42
```

## 📚 Citation

If you use this project in your research, please cite:

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

## 📝 License

This project follows the licensing terms of the original Clinical Behavioral Testing project.

## 🤝 Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

## 📧 Contact

For questions or issues, please:
1. Check the [documentation](docs/)
2. Search [existing issues](https://github.com/gyasifred/clinical-valence-testing/issues)
3. Open a new issue with detailed description

## 🔄 Version History

- **v1.0.0** (Current): Publication-ready release with full statistical analysis and visualization
  - ✅ All critical bugs fixed
  - ✅ Comprehensive documentation
  - ✅ Interactive visualizations
  - ✅ Statistical rigor for publication

---

**Status:** ✅ Ready for Publication

Built with ❤️ for the clinical NLP research community.
