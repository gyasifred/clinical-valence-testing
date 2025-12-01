# Clinical Valence Testing Project - Comprehensive Analysis Report

## Executive Summary

This project extends the Clinical Behavioral Testing framework by van Aken et al. to examine how valence-laden language (pejorative, laudatory, and neutral descriptors) affects clinical NLP model predictions, with a particular focus on attention pattern analysis in transformer-based diagnostic models.

**Project Status:** Active Development
**Primary Language:** Python
**ML Framework:** Transformers (Hugging Face), PyTorch
**Domain:** Clinical NLP, Medical Diagnosis Prediction
**Git Repository:** https://github.com/gyasifred/clinical-valence-testing

---

## 1. Project Overview

### 1.1 Purpose and Goals

The project investigates bias in clinical NLP models by:
- Testing how valence words (e.g., "compliant" vs. "difficult") affect diagnostic predictions
- Analyzing attention patterns to understand which words influence model decisions
- Providing a systematic framework for behavioral testing of clinical models
- Generating comprehensive statistical reports on model behavior under different linguistic conditions

### 1.2 Core Research Question

**How do subjective patient descriptors (pejorative, laudatory, neutral) influence diagnostic predictions made by transformer-based clinical NLP models?**

### 1.3 Key Innovation

Unlike the original behavioral testing framework, this project adds:
- **Attention analysis capabilities** - examining which words receive attention weights
- **Multiple valence categories** - testing pejorative, laudatory, neutral, and neutralized language
- **Comprehensive output tracking** - diagnosis probabilities, attention weights, and clinical notes
- **Statistical analysis tools** - visualization and reporting of shifts in predictions

---

## 2. Architecture and Components

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Main Entry Point                      │
│                          (main.py)                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ├──> Shift Transformations (test_shifts/)
                     │    ├── PejorativeShift
                     │    ├── LaudatoryShift
                     │    ├── NeutralValShift
                     │    └── NeutralizeShift
                     │
                     ├──> Behavioral Testing (valence_testing.py)
                     │    └── BehavioralTesting Framework
                     │
                     ├──> Model Predictions (prediction.py)
                     │    ├── TransformerPredictor (Base)
                     │    └── DiagnosisPredictor
                     │
                     ├──> Utilities (utils.py)
                     │    └── Pattern matching, data processing
                     │
                     └──> Analysis & Visualization
                          ├── plot.py (Diagnosis shifts)
                          └── plot_word_attention_shift.py (Attention analysis)
```

### 2.2 Core Components

#### A. Main Entry Point (`main.py`)
- Command-line interface using Fire library
- Orchestrates shift testing workflow
- Manages configuration and output directories
- Available shifts: neutralize, pejorative, laud, neutralval

#### B. Shift Transformations (`test_shifts/`)

**Base Architecture:**
- `BaseShift` - Abstract base class defining shift interface
- Common methods: `make_shift()`, `get_groups()`, `identify_group_in_text()`

**Shift Types:**

1. **PejorativeShift** - Negative patient descriptors
   - Levels: NON_COMPLIANT, UNCOOPERATIVE, RESISTANT, DIFFICULT
   - Terms: "non-compliant", "negligent", "difficult", "hostile", "manipulative", etc.
   - Inserts pejorative terms after patient characteristics

2. **LaudatoryShift** - Positive patient descriptors
   - Levels: COMPLIANT, COOPERATIVE, PLEASANT, RESPECTFUL
   - Terms: "compliant", "cooperative", "pleasant", "courteous", etc.
   - Inserts laudatory terms after patient characteristics

3. **NeutralValShift** - Neutral descriptors
   - Levels: NEUTRAL, NO_MENTION
   - Terms: "typical", "average", "presenting", "evaluated", etc.
   - Adds neutral, objective language

4. **NeutralizeShift** - Removes all valence words
   - Strips pejorative, laudatory, and neutral descriptors
   - Creates baseline for comparison

**Insertion Strategy:**
- Uses regex pattern matching to find patient characteristic positions
- Patterns include: age mentions, gender, patient status, ethnicity, medical conditions
- Intelligently places valence words after patient identifiers

#### C. Prediction System (`prediction.py`)

**Class Hierarchy:**
```
Predictor (Abstract)
  └── TransformerPredictor
        └── DiagnosisPredictor
```

**DiagnosisPredictor Features:**
- Multi-label classification (187 ICD codes with 100+ occurrences)
- Attention extraction from specific layer/head (default: layer 11, head 11)
- Batch processing with configurable batch size
- Checkpoint/recovery system for long-running predictions
- Three separate output streams:
  1. Diagnosis probabilities per ICD code
  2. Attention weights per word
  3. Clinical notes with valence labels

**Technical Details:**
- Model: AutoModelForSequenceClassification with attention outputs
- Tokenizer: BERT-based subword tokenization
- Attention aggregation: sum/average/max for subword tokens
- Probability calculation: Sigmoid activation for multi-label

**File Organization:**
```
results/
├── pejorative_TIMESTAMP_diagnosis.csv
├── pejorative_TIMESTAMP_attention.csv
├── pejorative_TIMESTAMP_clinical_notes.csv
├── laudatory_TIMESTAMP_diagnosis.csv
├── ...
└── checkpoints/
    └── checkpoint_TIMESTAMP.json
```

#### D. Behavioral Testing (`valence_testing.py`)

**BehavioralTesting Class:**
- Loads test dataset (CSV format)
- Applies shift transformations
- Manages prediction workflow
- Collects and returns statistics

**Error Handling:**
- Custom exceptions: `BehavioralTestingError`, `PredictionError`
- Input validation
- Graceful failure with informative messages

#### E. Utilities (`utils.py`)

**Key Functions:**

1. **`find_patient_characteristic_position_in_text()`**
   - Regex patterns for:
     - Age patterns (various formats: "45 yo", "45-year-old", etc.)
     - Gender identifiers
     - Patient types (inpatient, outpatient, etc.)
     - Ethnicity descriptors
     - Medical conditions
   - Returns insertion position for valence words
   - Cached using `@lru_cache` for performance

2. **`codes_that_occur_n_times_in_dataset()`**
   - Filters ICD codes by frequency
   - Returns top N most frequent codes
   - Used for multi-label classification setup

3. **Data processing utilities**
   - `sample_to_features()` - Tokenization
   - `collate_batch()` - Batch creation
   - `save_to_file()` - Result persistence

---

## 3. Data Flow and Workflow

### 3.1 Training Workflow (Jupyter Notebook)

```
1. Load CSV datasets (train/val/test)
2. Identify frequent ICD codes (n >= 100 occurrences)
3. Multi-hot encode labels (187 dimensions)
4. Initialize BioBERT model (CORe-clinical-outcome-biobert-v1)
5. Train with:
   - Batch size: 4
   - Learning rate: 2e-5
   - Epochs: 5
   - Optimizer: AdamW
   - Scheduler: Linear with warmup
6. Save best model checkpoint
```

### 3.2 Testing Workflow

```
Input: Test dataset CSV + Trained model checkpoint

Step 1: Load test data
        ↓
Step 2: Apply shift transformation
        ├── Pejorative groups
        ├── Laudatory groups
        ├── Neutral groups
        └── Neutralized versions
        ↓
Step 3: For each group:
        ├── Tokenize text
        ├── Run model inference
        ├── Extract attention weights
        ├── Calculate diagnosis probabilities
        └── Save to CSV files
        ↓
Step 4: Generate statistics
        └── Return shift summary
```

### 3.3 Analysis Workflow

```
Input: Result CSV files from testing

Step 1: Load all result files
        ├── *_diagnosis.csv
        ├── *_attention.csv
        └── *_clinical_notes.csv
        ↓
Step 2: Calculate shifts from baseline (neutralize)
        ├── Diagnosis probability shifts
        └── Attention weight shifts
        ↓
Step 3: Generate visualizations
        ├── Heatmaps of diagnosis shifts
        ├── Attention weight shift plots
        └── Distribution plots
        ↓
Step 4: Statistical analysis
        ├── Effect sizes per valence type
        ├── Most affected diagnoses
        └── Significant words
```

---

## 4. File Structure

```
clinical-valence-testing/
├── main.py                              # Entry point
├── prediction.py                        # Prediction classes
├── valence_testing.py                   # Testing framework
├── utils.py                             # Utility functions
├── __init__.py                          # Package initialization
│
├── test_shifts/                         # Shift implementations
│   ├── base_shift.py                   # Abstract base class
│   ├── pejorative_shift.py             # Negative descriptors
│   ├── laudatory_shift.py              # Positive descriptors
│   ├── neutralVal_shift.py             # Neutral descriptors
│   └── neutralize_shift.py             # Remove all valence
│
├── analysis_files/                      # Analysis scripts & results
│   ├── analysis.ipynb                  # Main analysis notebook
│   ├── plot.py                         # Diagnosis shift visualization
│   ├── plot_word_attention_shift.py    # Attention visualization
│   ├── output/                         # Generated plots
│   │   ├── valence_shifts.png
│   │   └── common_word_shifts.png
│   └── analysis_result/                # Analysis outputs
│       ├── full_analysis_report.md     # Statistical report
│       └── visualizations/             # Individual plots
│           ├── prediction_dist_*.png   # Per-diagnosis distributions
│           └── attention_heatmap.png   # Attention patterns
│
├── clinical_valence_testing_train_script.ipynb  # Training notebook
├── flowchart.png                        # System flowchart
├── README.md                            # Documentation
├── requirements.txt.txt                 # Dependencies
└── .gitignore                          # Git configuration
```

---

## 5. Technical Specifications

### 5.1 Dependencies

```
fire==0.7.0              # CLI interface
tqdm==2.2.3              # Progress bars
transformers==4.44.2     # Hugging Face models
numpy==1.21.0            # Numerical operations
pandas==1.2.5            # Data manipulation
torch                    # Deep learning (implied)
seaborn                  # Visualization (in analysis)
matplotlib               # Plotting (in analysis)
```

### 5.2 Model Configuration

**Base Model:** `bvanaken/CORe-clinical-outcome-biobert-v1`
- BioBERT variant trained on clinical outcomes
- BERT architecture with clinical domain adaptation

**Configuration:**
- Task: Multi-label classification
- Labels: Top 187 ICD codes (≥100 occurrences)
- Max sequence length: 512 tokens
- Attention: Extracted from layer 11, head 11
- Output: Sigmoid probabilities for each diagnosis

### 5.3 Data Format

**Input CSV Requirements:**
```
Columns:
- text: Clinical note text
- short_codes: Comma-separated ICD codes (e.g., "401,250,276")
```

**Output CSV Formats:**

1. **Diagnosis file:**
```
NoteID, Valence, Val_class, 401, 427, 276, ...(187 codes)
0, COMPLIANT, laudatory, 0.234, 0.012, 0.456, ...
```

2. **Attention file:**
```
NoteID, Word, AttentionWeight, Valence, Val_class
0, patient, 0.0234, COMPLIANT, laudatory
0, presents, 0.0123, COMPLIANT, laudatory
```

3. **Clinical notes file:**
```
NoteID, ClinicalNote, Valence, Val_class
0, "45 yo compliant male...", COMPLIANT, laudatory
```

### 5.4 Performance Considerations

**Checkpointing:**
- Saves state every 1000 samples (configurable)
- Enables recovery from interruptions
- JSON format with timestamp tracking

**Batch Processing:**
- Default batch size: 128
- Configurable per use case
- GPU support with automatic device detection

**Memory Management:**
- Thread-safe file handlers
- Streaming writes to CSV
- Proper resource cleanup

---

## 6. Key Findings and Capabilities

### 6.1 Analysis Capabilities

**Word Type Effects:**
- Effect sizes calculated for each valence word
- Example findings:
  - "resistant" (pejorative): +0.0045 effect
  - "compliant" (laudatory): +0.0024 effect
  - "neutral": +0.0007 effect

**Diagnosis Sensitivity:**
- Identifies which diagnoses are most affected by valence
- Compares probability distributions across valence groups
- Statistical summaries (mean, std, median, min, max)

**Attention Pattern Analysis:**
- Tracks which words receive highest attention
- Compares attention shifts across valence conditions
- Identifies common words with largest attention changes

### 6.2 Visualization Outputs

1. **Valence Shift Heatmaps**
   - Rows: Diagnoses (translated ICD codes)
   - Columns: Valence types (pejorative, laudatory, neutral)
   - Values: Probability shifts from baseline

2. **Attention Weight Heatmaps**
   - Top 30 common words
   - Attention weight changes by valence type
   - Color-coded diverging scale

3. **Distribution Plots**
   - Per-diagnosis probability distributions
   - Separated by valence group
   - Statistical comparisons

---

## 7. Usage Examples

### 7.1 Running Behavioral Tests

```bash
python main.py \
  --test_set_path /path/to/test.csv \
  --model_path /path/to/checkpoint \
  --shift_keys neutralize,pejorative,laud,neutralval \
  --task diagnosis \
  --save_dir ./results \
  --batch_size 128 \
  --head_num 11 \
  --layer_num 11
```

### 7.2 Generating Visualizations

```bash
# Diagnosis shifts
python analysis_files/plot.py \
  --results_dir ./results \
  --output_dir ./output \
  --baseline neutralize \
  --vmin -0.035 \
  --vmax 0.035

# Attention shifts
python analysis_files/plot_word_attention_shift.py
```

### 7.3 Programmatic Usage

```python
from valence_testing import BehavioralTesting
from prediction import DiagnosisPredictor
from test_shifts.pejorative_shift import PejorativeShift

# Initialize components
predictor = DiagnosisPredictor(
    checkpoint_path="model/checkpoint",
    test_set_path="data/test.csv",
    batch_size=128
)

bt = BehavioralTesting(test_dataset_path="data/test.csv")
shift = PejorativeShift()

# Run test
stats = bt.run_test(shift, predictor, "results/pejorative.csv")
```

---

## 8. Research Contributions

### 8.1 Methodological Innovations

1. **Systematic Valence Testing**
   - Four distinct shift types covering linguistic spectrum
   - Baseline neutralization for comparison
   - Automated insertion at appropriate text positions

2. **Multi-Level Analysis**
   - Diagnosis-level probability changes
   - Word-level attention patterns
   - Statistical effect quantification

3. **Reproducibility**
   - Checkpointing for long experiments
   - Deterministic shift application
   - Comprehensive logging

### 8.2 Potential Applications

1. **Bias Detection in Clinical AI**
   - Identify models sensitive to subjective language
   - Quantify bias magnitude
   - Guide debiasing efforts

2. **Model Robustness Testing**
   - Evaluate prediction stability
   - Test invariance to irrelevant features
   - Validate clinical decision support systems

3. **Explainability Research**
   - Understand attention patterns
   - Correlate language with predictions
   - Improve model transparency

---

## 9. Current Limitations and Future Work

### 9.1 Limitations

1. **Fixed Insertion Strategy**
   - Relies on regex patterns
   - May miss complex sentence structures
   - Limited to specific position in text

2. **Single Language**
   - English only
   - Clinical terminology specific to certain systems

3. **Model Dependency**
   - Designed for transformer models
   - Requires attention mechanism
   - BERT-specific tokenization

### 9.2 Future Enhancements

1. **Extended Shift Types**
   - Sentiment-based shifts
   - Socioeconomic descriptors
   - Cultural/ethnic bias testing

2. **Advanced Analysis**
   - Causal inference methods
   - Multi-layer attention analysis
   - Cross-model comparisons

3. **Automated Reporting**
   - Interactive dashboards
   - Real-time monitoring
   - Automated bias detection alerts

---

## 10. Development Status

### 10.1 Repository Information

- **Main Branch:** Currently on `claude/analyze-project-structure-01AQznp6tBZsJRknCkaYsu4j`
- **Recent Commits:**
  - Created using Colab
  - Updated flowchart
  - Modified output CSV columns (added Val_class)
  - Fixed argument parsing

### 10.2 Active Development

The project is under active development as indicated by:
- Recent commits improving functionality
- Iterative refinement of output format
- Bug fixes and enhancements

### 10.3 Production Readiness

**Strengths:**
- Robust error handling
- Checkpointing for reliability
- Comprehensive documentation
- Modular architecture

**Considerations:**
- Marked as "under active development"
- API may change
- Use with caution in production

---

## 11. Conclusion

This project represents a significant extension of clinical behavioral testing, adding attention analysis and comprehensive valence testing to understand how subjective language influences medical AI predictions. The modular architecture, robust implementation, and thorough analysis capabilities make it a valuable tool for bias detection and model evaluation in clinical NLP.

The system successfully combines:
- Linguistic transformations (shift framework)
- Deep learning inference (transformer models)
- Statistical analysis (effect quantification)
- Visualization (heatmaps, distributions)

This creates a complete pipeline for investigating and quantifying bias in clinical decision support systems.

---

**Report Generated:** 2025-12-01
**Project URL:** https://github.com/gyasifred/clinical-valence-testing
**License:** Following original Clinical Behavioral Testing project
