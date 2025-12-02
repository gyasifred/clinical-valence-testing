# Configuration Guide

Complete reference for configuring the Clinical Valence Testing framework.

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Configuration File Structure](#configuration-file-structure)
3. [Parameter Reference](#parameter-reference)
4. [Configuration Examples](#configuration-examples)
5. [Command-Line Override](#command-line-override)
6. [Environment Variables](#environment-variables)
7. [Advanced Configuration](#advanced-configuration)

---

## Configuration Overview

### Configuration Hierarchy

Parameters are resolved in the following order (highest to lowest priority):

1. **Command-line arguments** (highest priority)
2. **Configuration file** (YAML)
3. **Default values** (lowest priority)

**Example:**
```bash
# config.yaml has batch_size: 128
# Command-line specifies --batch_size 256
# Result: Uses 256 (command-line wins)

python main.py \
  --config_path ./config.yaml \
  --batch_size 256  # Overrides config file
```

### Configuration Files

**Default location:** `./config.yaml`

**Custom location:**
```bash
python main.py --config_path ./configs/my_experiment.yaml
```

---

## Configuration File Structure

### Complete config.yaml Template

```yaml
# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
model:
  # Model checkpoint path or HuggingFace model ID
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"

  # Batch size for inference
  # Larger = faster but needs more GPU memory
  # Recommended: GPU=128-512, CPU=32-64
  batch_size: 128

  # Use GPU for inference (true/false)
  use_gpu: true

  # Maximum sequence length (tokens)
  # Longer sequences = more memory
  max_length: 512

  # Attention analysis configuration
  attention:
    # Which transformer layer to analyze (0-11 for BioBERT)
    # Higher layers capture more semantic information
    layer_num: 11

    # Which attention head to analyze (0-11 for BioBERT)
    # Different heads focus on different patterns
    head_num: 11


# ============================================================================
# DATA CONFIGURATION
# ============================================================================
data:
  # Path to test dataset CSV file
  test_set_path: "./data/test_set.csv"

  # Column name containing clinical text
  text_column: "text"

  # Column name containing diagnosis codes
  code_column: "short_codes"

  # Maximum number of samples to process (null = all)
  max_samples: null

  # Minimum text length (characters) to include
  min_text_length: 50


# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================
prediction:
  # Task type (currently only "diagnosis" supported)
  task: "diagnosis"

  # Confidence threshold for predictions
  confidence_threshold: 0.5

  # Maximum number of diagnosis codes to predict
  max_predictions: 10

  # Save model predictions to disk
  save_predictions: true

  # Save attention weights to disk
  save_attention: true

  # Save checkpoint every N samples (for long runs)
  checkpoint_interval: 1000


# ============================================================================
# ANALYSIS CONFIGURATION
# ============================================================================
analysis:
  # Run statistical analysis on results
  run_statistical: true

  # Statistical significance level
  alpha: 0.05

  # Multiple comparison correction method
  # Options: "fdr" (FDR), "bonferroni", "none"
  correction_method: "fdr"

  # Bootstrap iterations for confidence intervals
  bootstrap_iterations: 1000

  # Confidence interval level
  confidence_level: 0.95

  # Effect size metrics to calculate
  # Options: "cohens_d", "hedges_g", "both"
  effect_size_metrics: "both"


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging:
  # Logging level
  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
  level: "INFO"

  # Enable console output
  console_output: true

  # Enable file output
  file_output: true

  # Log file path
  log_file: "./logs/experiment.log"

  # Use colored console output
  colored_output: true

  # Show progress bars for long operations
  progress_bars: true

  # Log format
  # Options: "detailed", "simple"
  log_format: "detailed"


# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
output:
  # Directory to save all results
  results_dir: "./results"

  # Create subdirectories for each experiment
  create_subdirs: true

  # Save format for results
  # Options: "csv", "json", "both"
  results_format: "csv"

  # Compression for large files
  # Options: "none", "gzip", "bz2"
  compression: "none"

  # Save diagnostic plots
  save_plots: true

  # Plot format
  # Options: "png", "pdf", "both"
  plot_format: "png"

  # Plot DPI (resolution)
  plot_dpi: 300


# ============================================================================
# SHIFT CONFIGURATION
# ============================================================================
shifts:
  # Which shifts to run
  # Options: "neutralize", "pejorative", "laud", "neutralval"
  # Use "all" to run all shifts
  enabled_shifts:
    - neutralize
    - pejorative
    - laud
    - neutralval

  # Shift-specific configurations
  pejorative:
    # Level-specific terms can be customized here
    enabled: true

  laudatory:
    enabled: true

  neutral_valence:
    enabled: true

  neutralize:
    enabled: true


# ============================================================================
# REPRODUCIBILITY CONFIGURATION
# ============================================================================
# Random seed for reproducibility
# Set to specific value (e.g., 42) for reproducible results
# Set to null for random behavior
random_seed: 42

# Enable PyTorch deterministic mode
# Ensures bit-exact reproducibility (may be slower)
deterministic: true

# Number of worker threads for data loading
num_workers: 4


# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================
performance:
  # Use mixed precision (FP16) training
  # Faster on modern GPUs, may slightly affect accuracy
  use_amp: false

  # Use gradient checkpointing (saves memory)
  gradient_checkpointing: false

  # Prefetch factor for data loading
  prefetch_factor: 2

  # Pin memory for faster GPU transfer
  pin_memory: true


# ============================================================================
# DEBUG CONFIGURATION
# ============================================================================
debug:
  # Enable debug mode (extra logging and checks)
  enabled: false

  # Save intermediate results
  save_intermediate: false

  # Profile performance
  profile: false

  # Verbose error messages
  verbose_errors: true
```

---

## Parameter Reference

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model.name` | string | required | Model checkpoint path or HuggingFace ID |
| `model.batch_size` | int | 128 | Batch size for inference |
| `model.use_gpu` | bool | false | Use GPU acceleration |
| `model.max_length` | int | 512 | Maximum sequence length |
| `model.attention.layer_num` | int | 11 | Transformer layer to analyze |
| `model.attention.head_num` | int | 11 | Attention head to analyze |

**Guidelines:**
- **batch_size**: Start with 128, increase if GPU memory allows
- **max_length**: 512 for most clinical texts, 128-256 for shorter texts
- **layer_num**: Higher layers (9-11) for semantic analysis, lower (0-3) for syntactic
- **head_num**: Experiment with different heads to find most sensitive

### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data.test_set_path` | string | required | Path to test CSV file |
| `data.text_column` | string | "text" | Column with clinical text |
| `data.code_column` | string | "short_codes" | Column with diagnosis codes |
| `data.max_samples` | int | null | Max samples to process (null=all) |
| `data.min_text_length` | int | 50 | Minimum text length (chars) |

**Data Format Requirements:**
```csv
text,short_codes
"Clinical text here...","['I21.9','R07.9']"
```

### Prediction Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prediction.task` | string | "diagnosis" | Prediction task type |
| `prediction.confidence_threshold` | float | 0.5 | Confidence threshold |
| `prediction.max_predictions` | int | 10 | Max predictions per sample |
| `prediction.save_predictions` | bool | true | Save predictions to disk |
| `prediction.save_attention` | bool | true | Save attention weights |
| `prediction.checkpoint_interval` | int | 1000 | Checkpoint frequency |

### Analysis Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `analysis.run_statistical` | bool | true | Run statistical analysis |
| `analysis.alpha` | float | 0.05 | Significance level |
| `analysis.correction_method` | string | "fdr" | Multiple comparison correction |
| `analysis.bootstrap_iterations` | int | 1000 | Bootstrap iterations |
| `analysis.confidence_level` | float | 0.95 | Confidence interval level |
| `analysis.effect_size_metrics` | string | "both" | Effect size metrics |

**correction_method options:**
- `"fdr"`: False Discovery Rate (Benjamini-Hochberg) - recommended for exploratory analysis
- `"bonferroni"`: Bonferroni correction - more conservative
- `"none"`: No correction - use with caution

### Logging Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `logging.level` | string | "INFO" | Log level |
| `logging.console_output` | bool | true | Print to console |
| `logging.file_output` | bool | true | Save to log file |
| `logging.log_file` | string | "./logs/..." | Log file path |
| `logging.colored_output` | bool | true | Use colors in console |
| `logging.progress_bars` | bool | true | Show progress bars |

**level options:**
- `"DEBUG"`: Detailed debugging information
- `"INFO"`: General information (recommended)
- `"WARNING"`: Only warnings and errors
- `"ERROR"`: Only errors

### Output Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output.results_dir` | string | "./results" | Results directory |
| `output.create_subdirs` | bool | true | Create experiment subdirs |
| `output.results_format` | string | "csv" | Output format |
| `output.compression` | string | "none" | File compression |
| `output.save_plots` | bool | true | Generate plots |
| `output.plot_format` | string | "png" | Plot file format |
| `output.plot_dpi` | int | 300 | Plot resolution |

### Reproducibility Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_seed` | int | 42 | Random seed |
| `deterministic` | bool | true | Deterministic mode |
| `num_workers` | int | 4 | Data loader workers |

**Important:** For exact reproducibility:
- Set `random_seed` to specific value (not null)
- Set `deterministic` to true
- Use same PyTorch/CUDA versions

---

## Configuration Examples

### Example 1: Quick Testing (Fast)

```yaml
# config_quick.yaml
model:
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"
  batch_size: 256
  use_gpu: true

data:
  test_set_path: "./data/test_set.csv"
  max_samples: 100  # Only 100 samples for quick test

analysis:
  run_statistical: false  # Skip statistical analysis

logging:
  level: "WARNING"  # Less verbose
  progress_bars: true

random_seed: 42
```

**Use case:** Quick sanity check before full run

### Example 2: Publication Quality (Slow but Thorough)

```yaml
# config_publication.yaml
model:
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"
  batch_size: 128
  use_gpu: true

data:
  test_set_path: "./data/full_test_set.csv"
  max_samples: null  # Process all samples

analysis:
  run_statistical: true
  correction_method: "bonferroni"  # Conservative
  bootstrap_iterations: 10000  # More iterations
  confidence_level: 0.99  # 99% CI

logging:
  level: "INFO"
  file_output: true
  log_file: "./logs/publication_experiment.log"

output:
  results_dir: "./results/publication"
  save_plots: true
  plot_format: "both"  # PNG and PDF
  plot_dpi: 600  # High resolution

random_seed: 12345  # Specific seed for paper
deterministic: true
```

**Use case:** Final experiments for publication

### Example 3: Large Dataset (Memory Optimized)

```yaml
# config_large.yaml
model:
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"
  batch_size: 64  # Smaller batches
  use_gpu: true
  max_length: 256  # Shorter sequences

data:
  test_set_path: "./data/large_dataset.csv"

prediction:
  checkpoint_interval: 5000  # Save progress frequently
  save_attention: false  # Skip attention to save space

output:
  compression: "gzip"  # Compress results

performance:
  use_amp: true  # Mixed precision
  gradient_checkpointing: true

random_seed: 42
```

**Use case:** Processing 100K+ samples

### Example 4: Debug Mode

```yaml
# config_debug.yaml
model:
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"
  batch_size: 8  # Very small batches
  use_gpu: false  # CPU for debugging

data:
  test_set_path: "./data/test_set.csv"
  max_samples: 10  # Just 10 samples

logging:
  level: "DEBUG"  # Maximum verbosity
  console_output: true
  colored_output: true

debug:
  enabled: true
  save_intermediate: true
  verbose_errors: true

random_seed: 42
```

**Use case:** Debugging code issues

### Example 5: Multi-GPU Setup

```yaml
# config_multi_gpu.yaml
model:
  name: "bvanaken/CORe-clinical-outcome-biobert-v1"
  batch_size: 512  # Large batches
  use_gpu: true

data:
  test_set_path: "./data/test_set.csv"

performance:
  use_amp: true
  pin_memory: true
  num_workers: 8

random_seed: 42
```

**Use case:** Maximum performance with multiple GPUs

---

## Command-Line Override

### Basic Override

Override any config parameter from command line:

```bash
python main.py \
  --config_path ./config.yaml \
  --batch_size 256 \
  --random_seed 999
```

### Complete Command-Line Reference

```bash
python main.py \
  # Required/common parameters
  --test_set_path ./data/test_set.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  \
  # Optional parameters
  --config_path ./config.yaml \
  --shift_keys pejorative,laud \
  --task diagnosis \
  --save_dir ./results \
  \
  # Performance parameters
  --gpu true \
  --batch_size 128 \
  \
  # Attention parameters
  --head_num 11 \
  --layer_num 11 \
  \
  # Data parameters
  --code_label short_codes \
  \
  # Analysis parameters
  --run_statistical_analysis true \
  \
  # Reproducibility
  --random_seed 42 \
  \
  # Checkpointing
  --checkpoint_interval 1000
```

### Parameter Priority Example

```yaml
# config.yaml
model:
  batch_size: 128
random_seed: 42
```

```bash
# Command line
python main.py \
  --config_path ./config.yaml \
  --batch_size 256  # Overrides config.yaml
  # random_seed uses 42 from config
```

**Result:**
- batch_size = 256 (from command line)
- random_seed = 42 (from config file)

---

## Environment Variables

### Supported Environment Variables

```bash
# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# PyTorch settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# HuggingFace cache
export TRANSFORMERS_CACHE=/path/to/cache

# Logging
export LOG_LEVEL=DEBUG
```

### Example Usage

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=1
python main.py --test_set_path ./data/test_set.csv

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1
python main.py --test_set_path ./data/test_set.csv

# Set cache directory
export TRANSFORMERS_CACHE=/scratch/hf_cache
python main.py --model_path bvanaken/CORe-clinical-outcome-biobert-v1
```

---

## Advanced Configuration

### Custom Configuration Classes

For programmatic configuration:

```python
from config_loader import Config, ModelConfig, DataConfig

# Create custom config
config = Config(
    model=ModelConfig(
        name="bvanaken/CORe-clinical-outcome-biobert-v1",
        batch_size=256,
        use_gpu=True
    ),
    data=DataConfig(
        test_set_path="./data/test_set.csv",
        text_column="text",
        code_column="short_codes"
    ),
    random_seed=42,
    deterministic=True
)

# Use config
from main import run
results = run(config_path=None, **config.__dict__)
```

### Configuration Validation

Validate config before running:

```python
from config_loader import get_config

# Load and validate
try:
    config = get_config("./config.yaml")
    print("✓ Configuration valid")
except Exception as e:
    print(f"✗ Configuration error: {e}")
```

### Dynamic Configuration

Modify config at runtime:

```python
from config_loader import get_config

# Load base config
config = get_config("./config.yaml")

# Modify for specific experiment
config.model.batch_size = 512
config.random_seed = 999
config.output.results_dir = "./results/experiment_42"

# Save modified config
import yaml
with open("./configs/modified_config.yaml", "w") as f:
    yaml.dump(config.__dict__, f)
```

---

## Configuration Best Practices

### ✅ DO:

1. **Use version control for configs**
```bash
git add config.yaml
git commit -m "Add experiment config"
```

2. **Name configs descriptively**
```
configs/
├── baseline_experiment.yaml
├── high_batch_publication.yaml
├── debug_small_sample.yaml
└── multi_gpu_production.yaml
```

3. **Document custom settings**
```yaml
# config.yaml
model:
  batch_size: 64  # Reduced for 8GB GPU memory
```

4. **Save config with results**
```bash
cp config.yaml ./results/experiment_1/config_used.yaml
```

### ❌ DON'T:

1. **Don't hardcode paths**
```yaml
# Bad
data:
  test_set_path: "/home/user/data/test.csv"

# Good
data:
  test_set_path: "./data/test.csv"
```

2. **Don't commit large files to config**
```yaml
# Bad - embedding in config
model:
  embedding: [0.1, 0.2, ..., 0.9]  # 1000s of values

# Good - reference file
model:
  embedding_path: "./embeddings/model_embedding.npy"
```

3. **Don't use inconsistent formats**
```yaml
# Be consistent with boolean values
use_gpu: true  # ✓ Good
use_amp: True  # ✓ Also fine
debug: yes     # ✗ Avoid mixing styles
```

---

## Troubleshooting Configuration

### Issue: Config Not Found

**Error:**
```
FileNotFoundError: config.yaml not found
```

**Solution:**
```bash
# Check current directory
pwd

# Create config
cp config.yaml.example config.yaml

# Or specify path
python main.py --config_path /full/path/to/config.yaml
```

### Issue: Invalid YAML Syntax

**Error:**
```
yaml.scanner.ScannerError: while scanning for the next token
```

**Solution:**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Common issues:
# - Missing quotes around strings with special chars
# - Inconsistent indentation (use spaces, not tabs)
# - Missing colons
```

### Issue: Type Mismatch

**Error:**
```
TypeError: batch_size must be int, got str
```

**Solution:**
```yaml
# Wrong
batch_size: "128"  # String

# Correct
batch_size: 128    # Integer
```

### Issue: Unknown Parameter

**Warning:**
```
Warning: Unknown config parameter 'batchsize'
```

**Solution:**
Check parameter spelling and structure:
```yaml
# Wrong
batchsize: 128

# Correct
model:
  batch_size: 128
```

---

## Summary

Key configuration concepts:

✅ **Hierarchy**: Command-line > Config file > Defaults
✅ **YAML Format**: Use proper syntax and types
✅ **Validation**: Check config before running experiments
✅ **Documentation**: Comment unusual settings
✅ **Version Control**: Track config changes
✅ **Reproducibility**: Set random_seed and deterministic mode

For more help:
- See `USAGE_GUIDE.md` for workflow examples
- See `README.md` for quick start
- Check example configs in `configs/` directory
