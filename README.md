# Clinical Valence Testing

Testing framework for analyzing how valence-laden language (pejorative, laudatory, neutral) affects clinical NLP model predictions and attention patterns.

## Overview

This framework performs behavioral testing of clinical NLP models by:
- Applying linguistic perturbations (valence shifts) to clinical texts
- Extracting model predictions and attention weights
- Performing statistical analysis of prediction changes
- Analyzing attention pattern shifts across different valence types

## Installation

### Prerequisites
- Python 3.8+
- NVIDIA H100 NVL GPU (95GB) with CUDA 12.1+

### Setup

```bash
# Clone repository
git clone https://github.com/gyasifred/clinical-valence-testing.git
cd clinical-valence-testing

# Run automated setup
bash setup_environment.sh

# Activate virtual environment
source venv/bin/activate
```

## Data Setup

Place your test data in the `data/` directory:

```bash
cp /path/to/DIA_GROUPS_3_DIGITS_adm_test.csv ./data/
cp /path/to/ALL_3_DIGIT_DIA_CODES.txt ./data/
```

**Expected CSV format:**
```csv
"id","text","short_codes"
"116159","CHIEF COMPLAINT: Positive ETT...","['I21.9','R07.9']"
```

## Usage

### Automated Run

```bash
bash run_analysis.sh
```

### Manual Run

```bash
python main.py \
  --test_set_path ./data/DIA_GROUPS_3_DIGITS_adm_test.csv \
  --model_path DATEXIS/CORe-clinical-diagnosis-prediction \
  --shift_keys neutralize,pejorative,laud,neutralval \
  --gpu true \
  --batch_size 768 \
  --code_label short_codes \
  --random_seed 42
```

## Configuration

Edit `config.yaml` for custom settings:

```yaml
model:
  name: "DATEXIS/CORe-clinical-diagnosis-prediction"  # Pre-trained ICD classifier
  batch_size: 768  # Optimized for H100 NVL (95GB); reduce to 512 for H100 SXM/PCIe (80GB)
  use_gpu: true
  device: "cuda"
  attention:
    layer_num: 11
    head_num: 11
    aggregation: "average"

data:
  test_set_path: "./data/DIA_GROUPS_3_DIGITS_adm_test.csv"
  text_label: "text"
  code_label: "short_codes"

random_seed: 42
```

## Shift Types

**Command:**
```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path DATEXIS/CORe-clinical-diagnosis-prediction \
  --save_dir ./results
```

**What happens:**
1. Configuration loaded from `config.yaml`
2. Random seed set to 42 (reproducible results)
3. All four shifts initialized (neutralize, pejorative, laud, neutralval)
4. Model loaded: `DATEXIS/CORe-clinical-diagnosis-prediction`
5. Behavioral testing runs for each shift:
   - Original samples loaded
   - Text perturbations applied
   - Model predictions generated
   - Attention weights extracted
   - Results compared across shifts
6. Statistical analysis performed
7. Results saved to `./results/`

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
  --model_path DATEXIS/CORe-clinical-diagnosis-prediction \
  --shift_keys pejorative \
  --save_dir ./results/pejorative_only
```

```bash
# Test pejorative and laudatory shifts
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path DATEXIS/CORe-clinical-diagnosis-prediction \
  --shift_keys pejorative,laud \
  --save_dir ./results/valence_comparison
```

### Step 5: Adjust Performance Parameters

**For H100 NVL GPU (95GB memory):**
```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path DATEXIS/CORe-clinical-diagnosis-prediction \
  --gpu true \
  --batch_size 768
```

**For H100 SXM/PCIe GPU (80GB memory):**
```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path DATEXIS/CORe-clinical-diagnosis-prediction \
  --gpu true \
  --batch_size 512
```

**For CPU-only systems (slower):**
```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path DATEXIS/CORe-clinical-diagnosis-prediction \
  --gpu false \
  --batch_size 32
```

### Step 6: Customize Attention Analysis

Analyze different attention heads and layers:

```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path DATEXIS/CORe-clinical-diagnosis-prediction \
  --layer_num 9 \
  --head_num 5
```

**Understanding attention parameters:**
- `layer_num`: Which transformer layer to analyze (0-11 for BioBERT)
- `head_num`: Which attention head to analyze (0-11 for BioBERT)
- `aggregation`: How to combine sub-token attention weights (default: "average")
  - **"average"** (recommended): Normalizes by number of sub-tokens, prevents bias toward longer words
  - **"sum"**: Adds all sub-token weights (can bias toward longer words)
  - **"max"**: Takes maximum sub-token weight
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
- Box plots comparing diagnosis probabilities across valence types
- Violin plots showing attention weight distributions
- Heatmaps of diagnosis probability shifts
- 3D surface plots of prediction changes

### Step 8: Run Statistical Analysis

The statistical analysis runs automatically, but you can customize it:

```bash
# Run without statistical analysis (faster)
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path DATEXIS/CORe-clinical-diagnosis-prediction \
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
  --model_path DATEXIS/CORe-clinical-diagnosis-prediction \
  --random_seed 123
```

**Reproducibility guarantees:**
- Same sample selection
- Same word insertions/removals
- Same model predictions (with deterministic mode)
- Same statistical results

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
| `batch_size` | int | 768 | Batch size for predictions (768 for H100 NVL, 512 for H100 SXM/PCIe) |
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

## Understanding the Output

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
results/
├── neutralize_shift_diagnosis.csv
├── pejorative_shift_diagnosis.csv
├── laud_shift_diagnosis.csv
├── neutralval_shift_diagnosis.csv
└── statistical_analysis.txt
```

Each CSV contains:
- Original and shifted text
- Model predictions before/after shift
- Prediction differences
- Attention weights

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test_set_path` | from config | Path to test CSV |
| `model_path` | from config | HuggingFace model ID or local path |
| `shift_keys` | all | Comma-separated shift names |
| `gpu` | false | Use GPU acceleration |
| `batch_size` | 768 | Batch size (768 for H100 NVL) |
| `code_label` | "short_codes" | Column name for diagnosis codes |
| `random_seed` | 42 | Random seed for reproducibility |

## GPU Optimization

| GPU Model | Memory | Batch Size |
|-----------|--------|------------|
| H100 NVL | 95GB | 768 |
| H100 SXM/PCIe | 80GB | 512 |
| A100 | 40-80GB | 256 |

## Project Structure

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

## Advanced Usage

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
      --model_path DATEXIS/CORe-clinical-diagnosis-prediction \
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

## Troubleshooting

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

## License

MIT License

## Version

1.0.0 (2025)
