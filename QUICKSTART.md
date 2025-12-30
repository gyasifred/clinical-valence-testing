# Quick Start Guide - Using Your Data Files

This guide shows you how to run the analysis with your specific data files:
- `DIA_GROUPS_3_DIGITS_adm_test.csv`
- `DIA_GROUPS_3_DIGITS_adm_train.csv`
- `DIA_GROUPS_3_DIGITS_adm_val.csv`
- `ALL_3_DIGIT_DIA_CODES.txt`

## Step-by-Step Instructions

### Step 1: Copy Your Data Files

From your current directory (where your data files are located), copy them to the project:

```bash
# Copy the test dataset
cp DIA_GROUPS_3_DIGITS_adm_test.csv /path/to/clinical-valence-testing/data/

# Copy the diagnosis codes reference
cp ALL_3_DIGIT_DIA_CODES.txt /path/to/clinical-valence-testing/data/

# (Optional) Copy training and validation sets if needed
cp DIA_GROUPS_3_DIGITS_adm_train.csv /path/to/clinical-valence-testing/data/
cp DIA_GROUPS_3_DIGITS_adm_val.csv /path/to/clinical-valence-testing/data/
```

### Step 2: Verify Your Data

```bash
cd /path/to/clinical-valence-testing

# Check files are in place
ls -lh data/

# View first few lines of test data
head -3 data/DIA_GROUPS_3_DIGITS_adm_test.csv

# Count number of test samples
tail -n +2 data/DIA_GROUPS_3_DIGITS_adm_test.csv | wc -l

# View diagnosis codes
head -10 data/ALL_3_DIGIT_DIA_CODES.txt
```

### Step 3: Run the Analysis (Automated)

The easiest way is to use the automated script:

```bash
bash run_analysis.sh
```

This script will:
1. Check that your data files exist
2. Validate the dataset format
3. Check Python environment
4. Detect GPU availability
5. Run all four shifts (neutralize, pejorative, laudatory, neutral valence)
6. Generate statistical analysis
7. Save results to `./results/run_YYYYMMDD_HHMMSS/`

### Step 4: Run the Analysis (Manual)

Or run manually with custom parameters:

```bash
# Basic run with all shifts
python main.py \
  --test_set_path ./data/DIA_GROUPS_3_DIGITS_adm_test.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --shift_keys neutralize,pejorative,laud,neutralval \
  --task diagnosis \
  --save_dir ./results \
  --code_label short_codes \
  --random_seed 42

# Run with H100 NVL GPU acceleration (95GB memory)
python main.py \
  --test_set_path ./data/DIA_GROUPS_3_DIGITS_adm_test.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --shift_keys neutralize,pejorative,laud,neutralval \
  --task diagnosis \
  --save_dir ./results \
  --gpu true \
  --batch_size 768 \
  --code_label short_codes \
  --random_seed 42

# Run specific shifts only
python main.py \
  --test_set_path ./data/DIA_GROUPS_3_DIGITS_adm_test.csv \
  --model_path bvanaken/CORe-clinical-outcome-biobert-v1 \
  --shift_keys pejorative,laud \
  --task diagnosis \
  --save_dir ./results/valence_only \
  --code_label short_codes
```

### Step 5: View Results

After the analysis completes, check your results:

```bash
# List all result files
ls -lh results/

# View statistical analysis
cat results/statistical_analysis.txt

# View individual shift statistics
cat results/neutralize_shift_diagnosis_stats.txt
cat results/pejorative_shift_diagnosis_stats.txt
cat results/laud_shift_diagnosis_stats.txt
cat results/neutralval_shift_diagnosis_stats.txt

# Examine CSV results (first 5 rows)
head -5 results/pejorative_shift_diagnosis.csv
```

## Your Data Format

Based on the files you showed, your data has this format:

### CSV Structure
```csv
"id","text","short_codes"
"116159","CHIEF COMPLAINT: Positive ETT...","['I21.9','R07.9']"
```

**Columns:**
- `id`: Unique identifier for each clinical note
- `text`: Full clinical text with:
  - Chief complaint
  - Present illness
  - Medical history
  - Medications
  - Allergies
- `short_codes`: List of diagnosis codes (ICD format)

### Diagnosis Codes File
The `ALL_3_DIGIT_DIA_CODES.txt` file contains all unique 3-digit diagnosis codes used across your datasets.

## Expected Output

After running the analysis, you'll get:

```
results/
├── neutralize_shift_diagnosis.csv        # Baseline results (valence removed)
├── neutralize_shift_diagnosis_stats.txt  # Statistics for baseline
├── pejorative_shift_diagnosis.csv        # Negative descriptors results
├── pejorative_shift_diagnosis_stats.txt  # Statistics for negative
├── laud_shift_diagnosis.csv              # Positive descriptors results
├── laud_shift_diagnosis_stats.txt        # Statistics for positive
├── neutralval_shift_diagnosis.csv        # Neutral descriptors results
├── neutralval_shift_diagnosis_stats.txt  # Statistics for neutral
└── statistical_analysis.txt              # Comprehensive statistical report
```

Each CSV contains:
- Original clinical text
- Shifted clinical text (with valence changes)
- Original predictions
- Shifted predictions
- Prediction differences
- Attention weights (if enabled)

## Dataset Statistics

Based on the file sizes you showed:

| File | Size | Estimated Samples |
|------|------|-------------------|
| DIA_GROUPS_3_DIGITS_adm_test.csv | 25 MB | ~3,000-5,000 |
| DIA_GROUPS_3_DIGITS_adm_train.csv | 87 MB | ~10,000-15,000 |
| DIA_GROUPS_3_DIGITS_adm_val.csv | 12 MB | ~1,500-2,500 |

**Processing Time Estimates:**
- With H100 NVL GPU (batch size 768): ~8-12 minutes for test set
- With other GPUs (batch size 256): ~15-20 minutes for test set
- Without GPU (batch size 32): ~30-45 minutes for test set

## Configuration for Your Data

The default `config.yaml` is already set up for your data format:

```yaml
data:
  test_set_path: "./data/DIA_GROUPS_3_DIGITS_adm_test.csv"
  text_label: "text"
  code_label: "short_codes"
```

## Troubleshooting

### Issue: "File not found" error
**Solution:** Make sure you copied the files to the `./data/` directory:
```bash
ls -l data/DIA_GROUPS_3_DIGITS_adm_test.csv
```

### Issue: "Out of memory" error
**Solution:** Reduce batch size:
```bash
python main.py --test_set_path ./data/DIA_GROUPS_3_DIGITS_adm_test.csv --batch_size 16 --gpu false
```

### Issue: Very slow processing
**Solution:**
1. Enable GPU if available: `--gpu true --batch_size 768` (for H100 NVL)
2. Or reduce test dataset for quick testing:
```bash
head -100 data/DIA_GROUPS_3_DIGITS_adm_test.csv > data/test_subset.csv
python main.py --test_set_path ./data/test_subset.csv
```

## Next Steps

After getting results:

1. **Review Statistical Analysis**: Check `statistical_analysis.txt` for significance tests and effect sizes

2. **Update Results Template**: Fill in the `results.txt` template with your actual values

3. **Analyze Specific Diagnoses**: Look for diagnosis codes most affected by valence shifts

4. **Generate Visualizations**: Use the interactive visualization tools to explore results

5. **Compare Across Datasets**: Run the same analysis on train/val sets to compare

## Questions?

See the main [README.md](README.md) for detailed documentation on:
- Configuration options
- Statistical methods
- Custom shifts
- Advanced usage
