# Data Directory

This directory should contain your test datasets for clinical valence testing.

## Required Files

### 1. Test Dataset (CSV)
Place your test dataset here. Based on your data:
- **File**: `DIA_GROUPS_3_DIGITS_adm_test.csv`
- **Format**: CSV with columns:
  - `id`: Unique identifier for each clinical note
  - `text`: Clinical text/notes
  - `short_codes`: Diagnosis codes (ICD codes)

**Example structure:**
```csv
"id","text","short_codes"
"116159","CHIEF COMPLAINT: Positive ETT...","['I21.9','R07.9']"
```

### 2. Diagnosis Codes Reference (Optional)
- **File**: `ALL_3_DIGIT_DIA_CODES.txt`
- **Purpose**: Complete list of all 3-digit diagnosis codes used in the dataset
- **Format**: One code per line or comma-separated

## How to Add Your Data

### Option 1: Copy Files Directly
```bash
# Copy your test dataset
cp /path/to/DIA_GROUPS_3_DIGITS_adm_test.csv ./data/

# Copy diagnosis codes reference
cp /path/to/ALL_3_DIGIT_DIA_CODES.txt ./data/
```

### Option 2: Create Symbolic Links
```bash
# Create symbolic link to test dataset
ln -s /path/to/DIA_GROUPS_3_DIGITS_adm_test.csv ./data/

# Create symbolic link to diagnosis codes
ln -s /path/to/ALL_3_DIGIT_DIA_CODES.txt ./data/
```

## Current Data Files

After adding your data, you should have:
```
data/
├── README.md (this file)
├── DIA_GROUPS_3_DIGITS_adm_test.csv (your test dataset)
└── ALL_3_DIGIT_DIA_CODES.txt (your diagnosis codes)
```

## Additional Datasets (Optional)

If you also want to use the training and validation sets:
- `DIA_GROUPS_3_DIGITS_adm_train.csv` - Training dataset
- `DIA_GROUPS_3_DIGITS_adm_val.csv` - Validation dataset

**Note**: The testing framework only requires the test dataset to run behavioral tests.
