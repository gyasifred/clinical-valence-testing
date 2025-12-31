# ICD Code Classification - Model Training Guide

## Overview

This guide explains how to train a BioBERT model for ICD code classification on MIMIC-III data. The trained model can then be used for clinical valence testing.

## Why Training is Needed

The base `bvanaken/CORe-clinical-outcome-biobert-v1` model is a **pre-trained language model**, NOT a trained ICD classifier. You need to:

1. **Fine-tune the model** on your MIMIC dataset with all 1,266 ICD codes
2. **Save the trained checkpoint** with proper label mappings
3. **Use the trained model** for behavioral testing

## Quick Start

### Option 1: Automated Training (Recommended)

```bash
./run_training.sh
```

This script will:
- Check for training/validation data
- Auto-configure batch size based on GPU
- Train the model for 5 epochs
- Save the trained model to `./models/icd_classifier_{timestamp}/final`

### Option 2: Manual Training

```bash
python train.py \
  --train_path ./data/DIA_GROUPS_3_DIGITS_adm_train.csv \
  --val_path ./data/DIA_GROUPS_3_DIGITS_adm_val.csv \
  --output_dir ./models/icd_classifier \
  --batch_size 32 \
  --num_epochs 5
```

## Data Requirements

### Expected Data Format

Your CSV file should have at least two columns:

```csv
text,short_codes
"Patient presents with chest pain...","401,786,V58"
"67 year old male with diabetes...","250,272,V45"
```

**Columns:**
- `text`: Clinical notes/discharge summaries
- `short_codes`: Comma-separated ICD-9-CM codes

### Data Files

The training script expects:

1. **Training data**: `./data/DIA_GROUPS_3_DIGITS_adm_train.csv`
2. **Validation data** (optional): `./data/DIA_GROUPS_3_DIGITS_adm_val.csv`
3. **Test data**: `./data/DIA_GROUPS_3_DIGITS_adm_test.csv`

If you only have test data, the script will automatically split it 80/20 for training/validation.

### Creating Train/Val/Test Splits

If you have a single dataset, split it first:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load full dataset
df = pd.read_csv('./data/DIA_GROUPS_3_DIGITS_adm_test.csv')

# Split: 70% train, 15% val, 15% test
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save splits
train_df.to_csv('./data/DIA_GROUPS_3_DIGITS_adm_train.csv', index=False)
val_df.to_csv('./data/DIA_GROUPS_3_DIGITS_adm_val.csv', index=False)
test_df.to_csv('./data/DIA_GROUPS_3_DIGITS_adm_test.csv', index=False)
```

## Training Configuration

### Hardware Requirements

**Recommended:**
- **GPU**: NVIDIA H100 (80-95GB) or A100 (40-80GB)
- **Batch size**: 32 (H100), 24 (A100), 16 (smaller GPUs)
- **Training time**: 2-4 hours for 50K samples (depends on GPU)

**Minimum:**
- **GPU**: Any CUDA-capable GPU with 16GB+ memory
- **Batch size**: 8-16
- **Training time**: 4-8 hours

**CPU-only** (not recommended):
- Very slow (10-20x slower than GPU)
- Use batch size 4-8

### Training Parameters

Edit `config.yaml` to customize training:

```yaml
training:
  num_epochs: 5              # Number of training epochs
  learning_rate: 2e-5        # Learning rate (AdamW)
  weight_decay: 0.01         # Weight decay for regularization
  warmup_steps: 0.1          # Warmup ratio (10% of total steps)
  scheduler: "linear"        # Learning rate scheduler
  early_stopping_patience: 3 # Stop if no improvement for 3 epochs
```

### Advanced Training Options

```bash
python train.py \
  --train_path ./data/train.csv \
  --val_path ./data/val.csv \
  --output_dir ./models/custom_model \
  --base_model bvanaken/CORe-clinical-outcome-biobert-v1 \
  --num_epochs 10 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --config_path ./configs/custom_config.yaml
```

## Training Process

The training script will:

1. **Load data** and filter codes (min 100 occurrences by default)
2. **Create multi-label encoding** for ICD codes
3. **Split data** (80/20 if no validation set provided)
4. **Initialize model** with correct number of labels (1,266 codes)
5. **Train** using AdamW optimizer with linear warmup
6. **Evaluate** after each epoch on validation set
7. **Save best model** based on F1-micro score
8. **Save final checkpoint** with all label mappings

## Output

After training completes, you'll find:

```
./models/icd_classifier_{timestamp}/
├── final/                      # Final trained model
│   ├── config.json            # Model config with label2id mapping
│   ├── pytorch_model.bin      # Trained weights
│   ├── tokenizer_config.json  # Tokenizer config
│   ├── vocab.txt              # Vocabulary
│   └── icd_codes.txt          # List of all ICD codes (1,266 codes)
├── checkpoint-{epoch}/         # Intermediate checkpoints
└── logs/                       # TensorBoard logs
```

## Using the Trained Model

### Update config.yaml

```yaml
model:
  name: "./models/icd_classifier_20251230_123456/final"
```

### Run Behavioral Testing

```bash
./run_analysis.sh
```

The script will now use your trained model instead of the base model.

### Verify Model Labels

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "./models/icd_classifier_20251230_123456/final"
)

print(f"Number of labels: {model.config.num_labels}")
print(f"Sample labels: {list(model.config.label2id.keys())[:10]}")
# Should show: ['401', '486', '582', ...]
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir ./models/icd_classifier_{timestamp}/logs
```

Then open http://localhost:6006 in your browser.

**Metrics to watch:**
- `f1_micro`: Overall F1 score (aim for >0.5)
- `f1_macro`: Per-code average F1
- `loss`: Training loss (should decrease)
- `eval_loss`: Validation loss (watch for overfitting)

### Training Logs

Check `./models/icd_classifier_{timestamp}/logs/` for detailed logs.

## Troubleshooting

### Out of Memory (OOM)

**Error:** `CUDA out of memory`

**Solution:**
```bash
# Reduce batch size
python train.py --batch_size 8

# Or use gradient accumulation
# Edit train.py: gradient_accumulation_steps=4
```

### Low F1 Score

**Problem:** F1 score < 0.3

**Solutions:**
1. **Train longer:** Increase epochs to 10
2. **Adjust learning rate:** Try 3e-5 or 1e-5
3. **Check data quality:** Ensure codes are clean
4. **Reduce min_frequency:** Include more codes (e.g., 50 instead of 100)

### Model Not Improving

**Problem:** Loss plateaus early

**Solutions:**
1. **Increase warmup:** Set `warmup_steps: 0.2` (20% warmup)
2. **Use smaller learning rate:** Try 1e-5
3. **Add dropout:** Edit model config (dropout: 0.2)
4. **Check data imbalance:** Some codes may be too rare

### Training Too Slow

**Solutions:**
1. **Use larger batch size** (if GPU allows)
2. **Enable mixed precision:** Already enabled with `fp16=True`
3. **Reduce max_length:** Set to 256 if notes are short
4. **Use fewer workers:** Reduce `dataloader_num_workers`

## Multi-GPU Training

For faster training with multiple GPUs:

```bash
# Use all available GPUs
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  train.py \
  --train_path ./data/train.csv \
  --batch_size 8  # Per GPU
```

## Transfer Learning

To fine-tune from a different base model:

```bash
python train.py \
  --train_path ./data/train.csv \
  --base_model emilyalsentzer/Bio_ClinicalBERT \
  --output_dir ./models/clinicalbert_icd
```

## Best Practices

1. **Always use validation data** for early stopping
2. **Monitor TensorBoard** during training
3. **Save multiple checkpoints** (already configured)
4. **Test on held-out data** before production use
5. **Document your hyperparameters** for reproducibility
6. **Use same preprocessing** for training and inference

## Citation

If you use this training pipeline, cite the original CORe paper:

```bibtex
@inproceedings{vanaken21,
  author    = {Betty van Aken and others},
  title     = {Clinical Outcome Prediction from Admission Notes using Self-Supervised Knowledge Integration},
  booktitle = {EACL 2021},
  year      = {2021},
}
```

## Support

For issues or questions:
1. Check logs in `./models/icd_classifier_{timestamp}/logs/`
2. Review TensorBoard metrics
3. Open an issue with error logs and configuration

---

**Status:** Production Ready

Successfully tested on MIMIC-III with 1,266 ICD codes.
