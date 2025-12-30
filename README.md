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
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
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
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"
  batch_size: 768  # H100 NVL (95GB)
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

| Shift | Description |
|-------|-------------|
| `neutralize` | Removes all valence terms (baseline) |
| `pejorative` | Adds negative descriptors (e.g., "difficult", "non-compliant") |
| `laud` | Adds positive descriptors (e.g., "cooperative", "pleasant") |
| `neutralval` | Adds neutral descriptors (e.g., "alert", "oriented") |

## Output

Results are saved to `./results/run_YYYYMMDD_HHMMSS/`:

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
├── main.py                  # Entry point
├── config.yaml             # Configuration
├── prediction.py           # Model inference
├── valence_testing.py      # Behavioral testing
├── statistical_analysis.py # Statistical tests
├── test_shifts/            # Shift implementations
├── data/                   # Test data
└── results/                # Output directory
```

## License

MIT License

## Version

1.0.0 (2025)
