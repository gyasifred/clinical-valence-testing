#!/bin/bash
################################################################################
# Clinical Valence Testing - Analysis Runner
################################################################################
# This script runs the clinical valence testing analysis using the provided
# DIA_GROUPS_3_DIGITS datasets and diagnosis codes.
#
# Usage:
#   1. Ensure your data files are in the ./data/ directory
#   2. Run: bash run_analysis.sh
#   3. Results will be saved to ./results/
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}Clinical Valence Testing - Analysis Runner${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Configuration
TEST_DATA="./data/DIA_GROUPS_3_DIGITS_adm_test.csv"
DIAGNOSIS_CODES="./data/ALL_3_DIGIT_DIA_CODES.txt"
MODEL_PATH="bvanaken/CORe-clinical-outcome-biobert-v1"
RESULTS_DIR="./results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_SUBDIR="${RESULTS_DIR}/run_${TIMESTAMP}"

# Check prerequisites
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check if test data exists
if [ ! -f "$TEST_DATA" ]; then
    echo -e "${RED}Error: Test dataset not found: $TEST_DATA${NC}"
    echo -e "${YELLOW}Please copy your data file to the ./data/ directory:${NC}"
    echo -e "  cp /path/to/DIA_GROUPS_3_DIGITS_adm_test.csv ./data/"
    exit 1
fi
echo -e "${GREEN}[OK] Test dataset found: $TEST_DATA${NC}"

# Check if diagnosis codes file exists (optional)
if [ -f "$DIAGNOSIS_CODES" ]; then
    echo -e "${GREEN}[OK] Diagnosis codes file found: $DIAGNOSIS_CODES${NC}"
    NUM_CODES=$(wc -w < "$DIAGNOSIS_CODES" 2>/dev/null || echo "unknown")
    echo -e "  Number of diagnosis codes: $NUM_CODES"
else
    echo -e "${YELLOW}[WARNING] Diagnosis codes file not found (optional): $DIAGNOSIS_CODES${NC}"
fi

# Check dataset format
echo ""
echo -e "${YELLOW}[2/6] Validating dataset format...${NC}"
HEAD_OUTPUT=$(head -3 "$TEST_DATA")
echo -e "${BLUE}First few lines of dataset:${NC}"
echo "$HEAD_OUTPUT"

# Count samples
NUM_SAMPLES=$(tail -n +2 "$TEST_DATA" | wc -l)
echo -e "${GREEN}[OK] Number of test samples: $NUM_SAMPLES${NC}"

# Check Python environment
echo ""
echo -e "${YELLOW}[3/6] Checking Python environment...${NC}"
if ! python -c "import torch, transformers, pandas, numpy" 2>/dev/null; then
    echo -e "${RED}Error: Required Python packages not installed${NC}"
    echo -e "${YELLOW}Please install dependencies:${NC}"
    echo -e "  pip install -r requirements.txt"
    exit 1
fi
echo -e "${GREEN}[OK] Python environment ready${NC}"

# Check GPU availability
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    USE_GPU="true"
    GPU_INFO=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEMORY=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / (1024**3))" 2>/dev/null)

    echo -e "${GREEN}[OK] GPU available: $GPU_INFO${NC}"
    echo -e "  GPU memory: ${GPU_MEMORY} GB"

    # Optimize batch size based on GPU memory
    if [[ "$GPU_INFO" == *"H100 NVL"* ]]; then
        BATCH_SIZE="768"  # H100 NVL has 95GB, can handle very large batches
        echo -e "${GREEN}  Using H100 NVL-optimized batch size: $BATCH_SIZE${NC}"
    elif [[ "$GPU_INFO" == *"H100"* ]]; then
        BATCH_SIZE="512"  # H100 SXM/PCIe has 80GB, can handle large batches
        echo -e "${GREEN}  Using H100-optimized batch size: $BATCH_SIZE${NC}"
    elif (( $(echo "$GPU_MEMORY > 40" | bc -l) )); then
        BATCH_SIZE="256"  # High-end GPU (A100, etc.)
        echo -e "${GREEN}  Using large batch size: $BATCH_SIZE${NC}"
    elif (( $(echo "$GPU_MEMORY > 16" | bc -l) )); then
        BATCH_SIZE="128"  # Mid-range GPU
        echo -e "${GREEN}  Using medium batch size: $BATCH_SIZE${NC}"
    else
        BATCH_SIZE="64"   # Lower-end GPU
        echo -e "${YELLOW}  Using small batch size: $BATCH_SIZE${NC}"
    fi
else
    USE_GPU="false"
    BATCH_SIZE="32"
    echo -e "${YELLOW}[WARNING] No GPU available, using CPU (slower)${NC}"
fi

# Create results directory
echo ""
echo -e "${YELLOW}[4/6] Preparing output directory...${NC}"
mkdir -p "$RESULTS_SUBDIR"
echo -e "${GREEN}[OK] Results will be saved to: $RESULTS_SUBDIR${NC}"

# Display configuration
echo ""
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}Configuration Summary${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo -e "Test Dataset:      $TEST_DATA"
echo -e "Model:             $MODEL_PATH"
echo -e "Results Directory: $RESULTS_SUBDIR"
echo -e "GPU Enabled:       $USE_GPU"
echo -e "Batch Size:        $BATCH_SIZE"
echo -e "Number of Samples: $NUM_SAMPLES"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Ask for confirmation
read -p "Do you want to proceed with the analysis? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Analysis cancelled.${NC}"
    exit 0
fi

# Run the analysis
echo ""
echo -e "${YELLOW}[5/6] Running clinical valence testing...${NC}"
echo -e "${BLUE}This may take a while depending on dataset size and hardware...${NC}"
echo ""

START_TIME=$(date +%s)

python main.py \
    --test_set_path "$TEST_DATA" \
    --model_path "$MODEL_PATH" \
    --shift_keys neutralize,pejorative,laud,neutralval \
    --task diagnosis \
    --save_dir "$RESULTS_SUBDIR" \
    --gpu "$USE_GPU" \
    --batch_size "$BATCH_SIZE" \
    --code_label "short_codes" \
    --random_seed 42 \
    --run_statistical_analysis true

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo -e "${GREEN}[SUCCESS] Analysis completed successfully${NC}"
echo -e "${GREEN}[INFO] Total time: ${MINUTES}m ${SECONDS}s${NC}"

# Display results summary
echo ""
echo -e "${YELLOW}[6/6] Results Summary${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}Generated Files:${NC}"
echo -e "${BLUE}================================================================================================${NC}"

if [ -d "$RESULTS_SUBDIR" ]; then
    ls -lh "$RESULTS_SUBDIR"

    echo ""
    echo -e "${BLUE}Key Output Files:${NC}"
    echo -e "  - ${GREEN}neutralize_shift_diagnosis.csv${NC} - Baseline (valence removed)"
    echo -e "  - ${GREEN}pejorative_shift_diagnosis.csv${NC} - Negative descriptors"
    echo -e "  - ${GREEN}laud_shift_diagnosis.csv${NC} - Positive descriptors"
    echo -e "  - ${GREEN}neutralval_shift_diagnosis.csv${NC} - Neutral descriptors"
    echo -e "  - ${GREEN}statistical_analysis.txt${NC} - Comprehensive statistical report"

    # Count results
    if [ -f "$RESULTS_SUBDIR/neutralize_shift_diagnosis.csv" ]; then
        RESULT_SAMPLES=$(tail -n +2 "$RESULTS_SUBDIR/neutralize_shift_diagnosis.csv" | wc -l)
        echo ""
        echo -e "${GREEN}[INFO] Processed $RESULT_SAMPLES samples${NC}"
    fi
fi

echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}[COMPLETE] Results saved to: $RESULTS_SUBDIR${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  1. Review the statistical analysis: cat $RESULTS_SUBDIR/statistical_analysis.txt"
echo -e "  2. Examine individual shift results in CSV files"
echo -e "  3. Update results.txt template with actual values"
echo ""
