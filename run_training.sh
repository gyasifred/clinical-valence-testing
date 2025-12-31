#!/bin/bash
################################################################################
# ICD Code Classification - Training Script
################################################################################
# This script trains a BioBERT model for ICD code classification on MIMIC data
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}ICD Code Classification - Model Training${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Configuration
TRAIN_DATA="./data/DIA_GROUPS_3_DIGITS_adm_train.csv"
VAL_DATA="./data/DIA_GROUPS_3_DIGITS_adm_val.csv"
OUTPUT_DIR="./models/icd_classifier"
BASE_MODEL="DATEXIS/CORe-clinical-diagnosis-prediction"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Check if training data exists
echo -e "${YELLOW}[1/5] Checking training data...${NC}"

if [ ! -f "$TRAIN_DATA" ]; then
    echo -e "${RED}Error: Training dataset not found: $TRAIN_DATA${NC}"
    echo -e "${YELLOW}Looking for test data to split...${NC}"

    # Check if test data exists
    TEST_DATA="./data/DIA_GROUPS_3_DIGITS_adm_test.csv"
    if [ -f "$TEST_DATA" ]; then
        echo -e "${YELLOW}Found test data at $TEST_DATA${NC}"
        echo -e "${YELLOW}Will use this for training (80/20 train/val split)${NC}"
        TRAIN_DATA="$TEST_DATA"
        VAL_DATA=""
    else
        echo -e "${RED}No data files found. Please provide training data.${NC}"
        echo -e "${YELLOW}Expected: $TRAIN_DATA or $TEST_DATA${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}[OK] Training data found: $TRAIN_DATA${NC}"

    if [ -f "$VAL_DATA" ]; then
        echo -e "${GREEN}[OK] Validation data found: $VAL_DATA${NC}"
    else
        echo -e "${YELLOW}[WARNING] No validation data. Will split training data 80/20.${NC}"
        VAL_DATA=""
    fi
fi

# Count samples
echo ""
echo -e "${YELLOW}[2/5] Analyzing dataset...${NC}"
NUM_SAMPLES=$(tail -n +2 "$TRAIN_DATA" | wc -l)
echo -e "  Training samples: $NUM_SAMPLES"

# Check Python environment
echo ""
echo -e "${YELLOW}[3/5] Checking Python environment...${NC}"
if ! python -c "import torch, transformers, sklearn" 2>/dev/null; then
    echo -e "${RED}Error: Required packages not installed${NC}"
    echo -e "${YELLOW}Install: pip install torch transformers scikit-learn${NC}"
    exit 1
fi
echo -e "${GREEN}[OK] Python environment ready${NC}"

# Check GPU
echo ""
echo -e "${YELLOW}[4/5] Checking hardware...${NC}"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_INFO=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    GPU_MEMORY=$(python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / (1024**3))" 2>/dev/null)

    echo -e "${GREEN}[OK] GPU available: $GPU_INFO${NC}"
    echo -e "  GPU memory: ${GPU_MEMORY} GB"

    # Optimize batch size based on GPU
    if [[ "$GPU_INFO" == *"H100"* ]]; then
        BATCH_SIZE="32"
        echo -e "${GREEN}  Using batch size: $BATCH_SIZE (H100 optimized)${NC}"
    elif (( $(echo "$GPU_MEMORY > 40" | bc -l) )); then
        BATCH_SIZE="24"
        echo -e "${GREEN}  Using batch size: $BATCH_SIZE${NC}"
    else
        BATCH_SIZE="16"
        echo -e "${YELLOW}  Using batch size: $BATCH_SIZE${NC}"
    fi
else
    BATCH_SIZE="8"
    echo -e "${YELLOW}[WARNING] No GPU available, using CPU (much slower)${NC}"
    echo -e "${YELLOW}  Using batch size: $BATCH_SIZE${NC}"
fi

# Display configuration
echo ""
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}Training Configuration${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo -e "Base Model:       $BASE_MODEL"
echo -e "Training Data:    $TRAIN_DATA"
if [ -n "$VAL_DATA" ]; then
    echo -e "Validation Data:  $VAL_DATA"
else
    echo -e "Validation Data:  Auto-split (20% of training)"
fi
echo -e "Output Directory: $OUTPUT_DIR"
echo -e "Batch Size:       $BATCH_SIZE"
echo -e "Number of Epochs: 5"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Ask for confirmation
read -p "Do you want to start training? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training cancelled.${NC}"
    exit 0
fi

# Run training
echo ""
echo -e "${YELLOW}[5/5] Starting model training...${NC}"
echo -e "${BLUE}This will take several hours depending on dataset size and hardware...${NC}"
echo ""

START_TIME=$(date +%s)

if [ -n "$VAL_DATA" ]; then
    python train.py \
        --train_path "$TRAIN_DATA" \
        --val_path "$VAL_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --base_model "$BASE_MODEL" \
        --batch_size "$BATCH_SIZE" \
        --num_epochs 5
else
    python train.py \
        --train_path "$TRAIN_DATA" \
        --output_dir "$OUTPUT_DIR" \
        --base_model "$BASE_MODEL" \
        --batch_size "$BATCH_SIZE" \
        --num_epochs 5
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo -e "${GREEN}================================================================================================${NC}"
echo -e "${GREEN}[SUCCESS] Training completed!${NC}"
echo -e "${GREEN}================================================================================================${NC}"
echo -e "Training time: ${HOURS}h ${MINUTES}m"
echo ""
echo -e "${BLUE}Model saved to: $OUTPUT_DIR${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Test the trained model:"
echo -e "     ${GREEN}./run_analysis.sh --model_path $OUTPUT_DIR/final${NC}"
echo ""
echo -e "  2. Update config.yaml with your trained model:"
echo -e "     ${GREEN}model:${NC}"
echo -e "     ${GREEN}  name: \"$OUTPUT_DIR/final\"${NC}"
echo ""
echo -e "${BLUE}================================================================================================${NC}"
