#!/bin/bash

#######################################################################
# One-Click Training Script for TS Prediction
# Usage: bash run_training.sh
#######################################################################

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════╗"
echo "║   Transition State Prediction - Training Pipeline     ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

# Step 1: Check Python environment
echo -e "${BLUE}[1/5] Checking Python environment...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Python not found!${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION}${NC}"

# Step 2: Check dependencies
echo -e "\n${BLUE}[2/5] Checking dependencies...${NC}"
python -c "
import sys
try:
    import torch
    import torch_geometric
    from rdkit import Chem
    import yaml
    print('✓ All core dependencies installed')
    print(f'  - PyTorch: {torch.__version__}')
    print(f'  - PyG: {torch_geometric.__version__}')
    print(f'  - CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  - GPU: {torch.cuda.get_device_name(0)}')
except ImportError as e:
    print(f'✗ Missing dependency: {e}')
    print('\\nPlease install dependencies:')
    print('  pip install -r requirements.txt')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Step 3: Check training data
echo -e "\n${BLUE}[3/5] Checking training data...${NC}"
if [ ! -d "./train_data" ]; then
    echo -e "${RED}✗ Training data not found!${NC}"
    echo -e "${YELLOW}Please ensure ./train_data directory exists${NC}"
    exit 1
fi

NUM_REACTIONS=$(find ./train_data -mindepth 1 -maxdepth 1 -type d | wc -l)
echo -e "${GREEN}✓ Found ${NUM_REACTIONS} reactions${NC}"

if [ "$NUM_REACTIONS" -lt 10 ]; then
    echo -e "${YELLOW}⚠ Warning: Very few training samples (<10)${NC}"
fi

# Step 4: Check configuration
echo -e "\n${BLUE}[4/5] Loading configuration...${NC}"
if [ ! -f "./config.yaml" ]; then
    echo -e "${RED}✗ config.yaml not found!${NC}"
    exit 1
fi

python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(f\"✓ Configuration loaded\")
print(f\"  - Epochs: {config['training']['epochs']}\")
print(f\"  - Batch size: {config['training']['batch_size']}\")
print(f\"  - Learning rate: {config['training']['learning_rate']}\")
print(f\"  - Mixed precision: {config['training']['mixed_precision']}\")
"

# Step 5: Start training
echo -e "\n${BLUE}[5/5] Starting training...${NC}"
echo "════════════════════════════════════════════════════════"
echo ""

# Create logs directory
mkdir -p logs

# Get timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Option: Run in background or foreground
read -p "Run in background? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Background mode
    nohup python -u train.py --config config.yaml > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > .training.pid
    
    echo -e "${GREEN}✓ Training started in background${NC}"
    echo ""
    echo "Process ID: $PID"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Monitor training:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "Stop training:"
    echo "  kill $PID"
    echo ""
    
    # Show initial logs
    sleep 2
    echo "Initial output:"
    echo "----------------------------------------"
    tail -20 "$LOG_FILE"
    echo "----------------------------------------"
else
    # Foreground mode
    python train.py --config config.yaml 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo -e "${GREEN}Training pipeline complete!${NC}"

