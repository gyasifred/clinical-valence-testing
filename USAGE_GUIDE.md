# Comprehensive Usage Guide

This guide provides detailed, step-by-step instructions for using the Clinical Valence Testing framework, from basic usage to advanced scenarios.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Workflows](#basic-workflows)
3. [Common Use Cases](#common-use-cases)
4. [Advanced Scenarios](#advanced-scenarios)
5. [Interpreting Results](#interpreting-results)
6. [Best Practices](#best-practices)
7. [Troubleshooting Guide](#troubleshooting-guide)

---

## Getting Started

### Prerequisites Checklist

Before running experiments, ensure you have:

- ✅ Python 3.8+ installed
- ✅ Virtual environment created and activated
- ✅ All dependencies installed (`pip install -r requirements.txt`)
- ✅ Test dataset prepared (CSV format with text and code columns)
- ✅ Model checkpoint or HuggingFace model ID
- ✅ Sufficient disk space for results (~ 1GB per 10K samples)
- ✅ (Optional) CUDA-capable GPU for faster processing

### Verify Your Setup

```bash
# 1. Check Python version
python --version  # Should be 3.8+

# 2. Check PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 3. Check CUDA availability (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 4. Verify all imports work
python -c "from config_loader import get_config; from logger import setup_logging; print('✓ All imports successful')"
```

---

## Basic Workflows

### Workflow 1: First-Time User - Running All Shifts

**Goal:** Run all shifts on your dataset to understand overall valence effects.

**Steps:**

1. **Prepare your data** (`./data/test_set.csv`):
```csv
text,short_codes
"Patient presents with acute chest pain. Alert and oriented x3.","['I21.9','R07.9']"
"58 yo male with history of diabetes. Patient is cooperative.","['E11.9','Z79.4']"
```

2. **Create minimal config** (`config.yaml`):
```yaml
model:
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"
  batch_size: 128
  use_gpu: true

data:
  test_set_path: "./data/test_set.csv"

output:
  results_dir: "./results"

random_seed: 42
deterministic: true
```

3. **Run the experiment**:
```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1
```

4. **Expected console output**:
```
[INFO] Configuration loaded successfully
[INFO] Random seed set to 42
[INFO] Deterministic mode enabled
[INFO] Running with shifts: ['neutralize', 'pejorative', 'laud', 'neutralval']
[INFO] Initializing diagnosis predictor...
[INFO] Model loaded: bvanaken/CORe-clinical-outcome-biobert-v1
[INFO] Initializing behavioral testing framework...
[INFO] Results will be saved to: ./results
[INFO] Starting neutralize shift testing...
[INFO] Completed neutralize shift testing
[INFO] Results saved to ./results/neutralize_shift_diagnosis.csv
...
[INFO] Running statistical analysis...
[INFO] Statistical analysis saved to ./results/statistical_analysis.txt
[INFO] All behavioral tests completed successfully!
```

5. **Check results**:
```bash
ls -lh ./results/
```

**Expected files:**
```
neutralize_shift_diagnosis.csv
neutralize_shift_diagnosis_stats.txt
pejorative_shift_diagnosis.csv
pejorative_shift_diagnosis_stats.txt
laud_shift_diagnosis.csv
laud_shift_diagnosis_stats.txt
neutralval_shift_diagnosis.csv
neutralval_shift_diagnosis_stats.txt
statistical_analysis.txt
```

### Workflow 2: Testing Specific Hypothesis

**Goal:** Test if pejorative language increases psychiatric diagnoses.

**Steps:**

1. **Run only pejorative shift**:
```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --shift_keys pejorative \
  --save_dir ./results/pejorative_psychiatric
```

2. **Analyze results** (see [Interpreting Results](#interpreting-results))

3. **Run baseline (neutralize) for comparison**:
```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --shift_keys neutralize \
  --save_dir ./results/neutralize_baseline
```

4. **Compare results**:
```python
import pandas as pd

# Load results
pejorative = pd.read_csv('./results/pejorative_psychiatric/pejorative_shift_diagnosis.csv')
neutralize = pd.read_csv('./results/neutralize_baseline/neutralize_shift_diagnosis.csv')

# Compare prediction differences
print(f"Pejorative avg change: {pejorative['prediction_diff'].mean():.4f}")
print(f"Neutralize avg change: {neutralize['prediction_diff'].mean():.4f}")
```

### Workflow 3: GPU vs CPU Comparison

**Goal:** Compare processing time between GPU and CPU.

**Steps:**

1. **Run with GPU**:
```bash
time python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --gpu true \
  --batch_size 256 \
  --save_dir ./results/gpu_run
```

2. **Run with CPU**:
```bash
time python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --gpu false \
  --batch_size 32 \
  --save_dir ./results/cpu_run
```

3. **Compare times**:
```
GPU: 2m 15s
CPU: 18m 42s
Speedup: 8.3x
```

---

## Common Use Cases

### Use Case 1: Reproducibility for Publication

**Scenario:** You need exactly reproducible results for your paper.

**Solution:**

```bash
# Set explicit random seed
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --random_seed 12345 \
  --save_dir ./results/paper_experiment_1
```

**Verification:**
```bash
# Run again with same seed
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --random_seed 12345 \
  --save_dir ./results/paper_experiment_1_verify

# Compare results (should be identical)
diff ./results/paper_experiment_1/pejorative_shift_diagnosis.csv \
     ./results/paper_experiment_1_verify/pejorative_shift_diagnosis.csv
```

### Use Case 2: Large Dataset Processing

**Scenario:** You have a large dataset (100K+ samples) and want to process it efficiently.

**Solution:**

```yaml
# config_large.yaml
model:
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"
  batch_size: 512        # Larger batches
  use_gpu: true

prediction:
  checkpoint_interval: 5000  # Save progress every 5K samples

logging:
  level: "INFO"
  progress_bar: true
```

```bash
python main.py \
  --config_path ./config_large.yaml \
  --test_set_path ./data/large_test_set.csv \
  --save_dir ./results/large_experiment
```

**Monitoring:**
```bash
# Watch progress in real-time
tail -f ./logs/experiment.log
```

### Use Case 3: Multiple Models Comparison

**Scenario:** Compare how different clinical models respond to valence shifts.

**Script:** `compare_models.sh`
```bash
#!/bin/bash

models=(
  "bvanaken/CORe-clinical-outcome-biobert-v1"
  "emilyalsentzer/Bio_ClinicalBERT"
  "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)

for model in "${models[@]}"; do
  model_name=$(basename $model)
  echo "Testing model: $model_name"

  python main.py \
    --test_set_path ./data/test_set.csv \
    --model_path "$model" \
    --save_dir "./results/model_comparison/$model_name"
done

echo "All models tested. Results in ./results/model_comparison/"
```

**Run:**
```bash
chmod +x compare_models.sh
./compare_models.sh
```

### Use Case 4: Attention Head Analysis

**Scenario:** Determine which attention heads are most sensitive to valence language.

**Solution:**

```bash
#!/bin/bash
# analyze_attention_heads.sh

for layer in {9..11}; do
  for head in {0..11}; do
    echo "Testing layer $layer, head $head"

    python main.py \
      --test_set_path ./data/test_set.csv \
      --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
      --shift_keys pejorative \
      --layer_num $layer \
      --head_num $head \
      --save_dir "./results/attention_analysis/layer${layer}_head${head}"
  done
done
```

**Analysis:**
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Collect results
results = []
for layer in range(9, 12):
    for head in range(12):
        path = f"./results/attention_analysis/layer{layer}_head{head}/pejorative_shift_diagnosis.csv"
        df = pd.read_csv(path)
        avg_change = df['prediction_diff'].mean()
        results.append({'layer': layer, 'head': head, 'avg_change': avg_change})

df_results = pd.DataFrame(results)

# Create heatmap
pivot = df_results.pivot(index='layer', columns='head', values='avg_change')
plt.figure(figsize=(12, 4))
plt.imshow(pivot, cmap='RdYlBu_r', aspect='auto')
plt.colorbar(label='Average Prediction Change')
plt.xlabel('Attention Head')
plt.ylabel('Layer')
plt.title('Attention Head Sensitivity to Pejorative Language')
plt.savefig('./results/attention_heatmap.png', dpi=300)
```

### Use Case 5: Subset Analysis

**Scenario:** Analyze impact on specific patient subgroups (e.g., diabetes patients).

**Solution:**

```python
# filter_and_analyze.py
import pandas as pd

# Load original dataset
df = pd.read_csv('./data/test_set.csv')

# Filter for diabetes patients (ICD code E11.*)
diabetes_df = df[df['short_codes'].str.contains('E11')]
diabetes_df.to_csv('./data/diabetes_subset.csv', index=False)

print(f"Diabetes subset: {len(diabetes_df)} samples")
```

```bash
# Run analysis on subset
python main.py \
  --test_set_path ./data/diabetes_subset.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --save_dir ./results/diabetes_analysis
```

---

## Advanced Scenarios

### Scenario 1: Custom Shift Implementation

**Goal:** Create a custom shift for age-related language.

**Implementation:**

```python
# test_shifts/age_shift.py
import re
import random
import logging
from enum import Enum
from typing import Optional
from test_shifts.base_shift import BaseShift

logger = logging.getLogger(__name__)

class AgeDescriptor(Enum):
    ELDERLY = 1
    YOUNG = 2
    NO_MENTION = 3

class AgeShift(BaseShift):
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)

        self.age_terms = {
            AgeDescriptor.ELDERLY: ["elderly", "geriatric", "aged", "senior"],
            AgeDescriptor.YOUNG: ["young", "youthful", "juvenile"]
        }

        logger.info(f"AgeShift initialized with random_seed={random_seed}")

    def get_groups(self):
        return [AgeDescriptor.ELDERLY, AgeDescriptor.YOUNG]

    def get_group_names(self):
        return [group.name for group in self.get_groups()]

    def identify_group_in_text(self, text: str) -> Optional[AgeDescriptor]:
        text_lower = text.lower()

        for term in self.age_terms[AgeDescriptor.ELDERLY]:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                return AgeDescriptor.ELDERLY

        for term in self.age_terms[AgeDescriptor.YOUNG]:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                return AgeDescriptor.YOUNG

        return None

    def get_shift_method(self, sample: str, group: AgeDescriptor) -> str:
        # Insert age descriptor if not present
        current_group = self.identify_group_in_text(sample)

        if current_group == group:
            return sample  # Already has this descriptor

        # Remove existing age descriptors
        shifted_text = sample
        for age_type, terms in self.age_terms.items():
            for term in terms:
                pattern = r'\b' + re.escape(term) + r'\b'
                shifted_text = re.sub(pattern, '', shifted_text, flags=re.IGNORECASE)

        # Insert new descriptor
        term_to_insert = random.choice(self.age_terms[group])

        # Find insertion point (after "patient" mention)
        patient_match = re.search(r'\bpatient\b', shifted_text, re.IGNORECASE)
        if patient_match:
            insert_pos = patient_match.end()
            shifted_text = (
                shifted_text[:insert_pos] +
                f" is {term_to_insert}" +
                shifted_text[insert_pos:]
            )

        # Clean whitespace
        shifted_text = re.sub(r'\s+', ' ', shifted_text).strip()

        logger.debug(f"Applied {group.name} shift")
        return shifted_text
```

**Register in main.py:**

```python
# Add to imports
from test_shifts.age_shift import AgeShift

# Update get_shift_map function
def get_shift_map(random_seed: Optional[int] = None):
    return {
        "neutralize": NeutralizeShift(random_seed=random_seed),
        "pejorative": PejorativeShift(random_seed=random_seed),
        "laud": LaudatoryShift(random_seed=random_seed),
        "neutralval": NeutralValShift(random_seed=random_seed),
        "age": AgeShift(random_seed=random_seed)  # Add custom shift
    }
```

**Use custom shift:**
```bash
python main.py \
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --shift_keys age \
  --save_dir ./results/age_analysis
```

### Scenario 2: Parallel Experiments

**Goal:** Run multiple experiments simultaneously.

**Script:** `parallel_experiments.py`
```python
import subprocess
import concurrent.futures

experiments = [
    {
        'name': 'exp1_pejorative',
        'shift_keys': 'pejorative',
        'random_seed': 42
    },
    {
        'name': 'exp2_laudatory',
        'shift_keys': 'laud',
        'random_seed': 42
    },
    {
        'name': 'exp3_neutral',
        'shift_keys': 'neutralval',
        'random_seed': 42
    }
]

def run_experiment(exp):
    cmd = [
        'python', 'main.py',
        '--test_set_path', './data/test_set.csv',
        '--model_path', 'bvanaken/CORe-clinical-outcome-biobert-v1',
        '--shift_keys', exp['shift_keys'],
        '--random_seed', str(exp['random_seed']),
        '--save_dir', f"./results/{exp['name']}"
    ]

    print(f"Starting {exp['name']}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(f"Completed {exp['name']}")
    return exp['name'], result.returncode

# Run experiments in parallel
with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(run_experiment, exp) for exp in experiments]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

print("All experiments completed:")
for name, returncode in results:
    status = "✓" if returncode == 0 else "✗"
    print(f"  {status} {name}")
```

**Run:**
```bash
python parallel_experiments.py
```

---

## Interpreting Results

### Understanding CSV Outputs

**Structure:**
```python
import pandas as pd

# Load results
df = pd.read_csv('./results/pejorative_shift_diagnosis.csv')

# Display structure
print(df.columns)
# Index(['sample_id', 'original_text', 'shifted_text',
#        'original_prediction', 'shifted_prediction',
#        'prediction_diff', 'attention_weights'], dtype='object')

# Example row
print(df.iloc[0])
```

**Key Metrics:**

1. **Prediction Difference** (`prediction_diff`):
   - Range: 0.0 to 1.0
   - Interpretation:
     - < 0.05: Minimal change
     - 0.05-0.15: Small change
     - 0.15-0.30: Medium change
     - > 0.30: Large change

2. **Sample Statistics:**
```python
# Calculate statistics
print(f"Mean change: {df['prediction_diff'].mean():.4f}")
print(f"Median change: {df['prediction_diff'].median():.4f}")
print(f"Std dev: {df['prediction_diff'].std():.4f}")
print(f"Max change: {df['prediction_diff'].max():.4f}")

# Percentage with significant change (> 0.15)
significant_change = (df['prediction_diff'] > 0.15).sum()
print(f"Samples with significant change: {significant_change} ({significant_change/len(df)*100:.1f}%)")
```

### Analyzing Statistical Reports

**Structure of `statistical_analysis.txt`:**

```
COMPREHENSIVE STATISTICAL ANALYSIS
===================================

1. DESCRIPTIVE STATISTICS
-------------------------
Shift: pejorative
  Mean change: 0.085
  Median change: 0.063
  Std dev: 0.094

2. HYPOTHESIS TESTS
-------------------
Paired t-test (pejorative vs baseline):
  t-statistic: 5.23
  p-value: 0.0001 ***
  Significant: Yes

Wilcoxon signed-rank test:
  statistic: 123456.0
  p-value: 0.0002 ***

3. EFFECT SIZES
---------------
Cohen's d: 0.34 (medium effect)
Hedges' g: 0.33 (medium effect)
95% CI: [0.052, 0.118]

4. MULTIPLE COMPARISON CORRECTION
----------------------------------
FDR-corrected p-values:
  pejorative vs baseline: 0.0002 ***
  laudatory vs baseline: 0.023 *

Bonferroni-corrected p-values:
  pejorative vs baseline: 0.0006 ***
  laudatory vs baseline: 0.069 ns
```

**Interpretation Guide:**

- **p-value interpretation:**
  - p < 0.001: *** (highly significant)
  - p < 0.01: ** (very significant)
  - p < 0.05: * (significant)
  - p >= 0.05: ns (not significant)

- **Effect size interpretation (Cohen's d):**
  - |d| < 0.2: Small effect
  - 0.2 ≤ |d| < 0.5: Medium effect
  - |d| ≥ 0.5: Large effect

### Visualizing Results

**Create comparison plots:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load all results
shifts = ['pejorative', 'laud', 'neutralval']
data = []

for shift in shifts:
    df = pd.read_csv(f'./results/{shift}_shift_diagnosis.csv')
    df['shift_type'] = shift
    data.append(df)

combined = pd.concat(data)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Box plot
sns.boxplot(data=combined, x='shift_type', y='prediction_diff', ax=axes[0])
axes[0].set_title('Prediction Changes by Shift Type')
axes[0].set_ylabel('Prediction Difference')
axes[0].set_xlabel('Shift Type')

# Distribution plot
for shift in shifts:
    data_subset = combined[combined['shift_type'] == shift]['prediction_diff']
    sns.kdeplot(data=data_subset, label=shift, ax=axes[1])

axes[1].set_title('Distribution of Prediction Changes')
axes[1].set_xlabel('Prediction Difference')
axes[1].set_ylabel('Density')
axes[1].legend()

plt.tight_layout()
plt.savefig('./results/comparison_plots.png', dpi=300)
print("Plots saved to ./results/comparison_plots.png")
```

---

## Best Practices

### 1. Reproducibility

✅ **DO:**
- Always set a random seed
- Document your Python/PyTorch versions
- Save your config files with results
- Use version control for code

```bash
# Save environment info
pip freeze > ./results/requirements_used.txt
python --version > ./results/python_version.txt
git rev-parse HEAD > ./results/git_commit.txt
```

❌ **DON'T:**
- Run experiments without setting random seed
- Modify code between runs without documentation
- Delete intermediate results

### 2. Performance Optimization

✅ **DO:**
- Use GPU when available
- Adjust batch size based on GPU memory
- Save checkpoints for long runs
- Monitor system resources

```python
# Monitor GPU usage
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True)
print(result.stdout.decode())
```

❌ **DON'T:**
- Use batch_size > GPU memory capacity
- Run CPU experiments overnight when GPU available
- Ignore checkpoint_interval for large datasets

### 3. Data Management

✅ **DO:**
- Keep raw data separate from results
- Use descriptive directory names
- Document data preprocessing steps
- Validate data before running experiments

```bash
# Organize data
./data/
  ├── raw/
  │   └── original_data.csv
  ├── processed/
  │   └── test_set.csv
  └── README.md  # Document preprocessing
```

❌ **DON'T:**
- Overwrite original data
- Mix test and training data
- Forget to document exclusion criteria

---

## Troubleshooting Guide

### Problem 1: ImportError

**Symptom:**
```
ImportError: No module named 'transformers'
```

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# If still fails, try upgrading pip
pip install --upgrade pip
pip install -r requirements.txt
```

### Problem 2: CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solution:**
```bash
# Reduce batch size
python main.py --batch_size 32  # Try 32, 16, or 8

# Or use CPU
python main.py --gpu false
```

### Problem 3: Slow Processing

**Symptom:**
Processing 1000 samples takes > 30 minutes

**Diagnosis:**
```python
# Check if GPU is being used
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")
```

**Solution:**
```bash
# Enable GPU
python main.py --gpu true

# Increase batch size
python main.py --gpu true --batch_size 256
```

### Problem 4: Inconsistent Results

**Symptom:**
Running same command gives different results

**Solution:**
```bash
# Set explicit random seed
python main.py --random_seed 42

# Verify deterministic mode in config.yaml
deterministic: true
```

### Problem 5: Empty Results

**Symptom:**
CSV files are created but empty or with only headers

**Diagnosis:**
Check logs:
```bash
tail -50 ./logs/experiment.log
```

**Common causes:**
1. No samples match shift criteria
2. Data format mismatch
3. Model loading failure

**Solution:**
```python
# Validate data format
import pandas as pd
df = pd.read_csv('./data/test_set.csv')
print(df.head())
print(df.columns)
print(df['text'].iloc[0])  # Check text format
print(df['short_codes'].iloc[0])  # Check codes format
```

---

## Summary

This guide covered:

✅ Basic workflows for first-time users
✅ Common use cases for research
✅ Advanced scenarios and customization
✅ Result interpretation and visualization
✅ Best practices for reproducibility
✅ Troubleshooting common issues

For more information:
- See `README.md` for quick reference
- See `CONFIGURATION_GUIDE.md` for config details
- See `API_REFERENCE.md` for programmatic usage
- Check GitHub issues for known problems

**Need help?** Open an issue with:
1. Your command
2. Error message
3. Python/PyTorch versions
4. System info (OS, GPU)
