#!/bin/bash
# Quick Test Script for SRDownscalling
# Tests model architecture and data loading with limited samples

set -e

echo "========================================"
echo "SRDownscalling - Quick Test"
echo "========================================"

# Activate virtual environment if exists
if [ -f "/home/oriol/.openclaw/workspace/venv_sr_test/bin/activate" ]; then
    source /home/oriol/.openclaw/workspace/venv_sr_test/bin/activate
fi

# Configuration
LR_DIR="/home/oriol/data/WRF/1469893/d01"
HR_DIR="/home/oriol/data/WRF/1469893/d05"
OUTPUT_DIR="/home/oriol/.openclaw/workspace/git/SRDownscalling/outputs"
SAMPLES=5
EPOCHS=2
BATCH_SIZE=2

echo ""
echo "Configuration:"
echo "  LR Dir: $LR_DIR"
echo "  HR Dir: $HR_DIR"
echo "  Samples: $SAMPLES"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Check data exists
if [ ! -d "$LR_DIR" ] || [ ! -d "$HR_DIR" ]; then
    echo "ERROR: Data directories not found!"
    exit 1
fi

# Count files
LR_COUNT=$(ls -1 "$LR_DIR"/*.nc 2>/dev/null | wc -l)
HR_COUNT=$(ls -1 "$HR_DIR"/*.nc 2>/dev/null | wc -l)
echo "Files found: LR=$LR_COUNT, HR=$HR_COUNT"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run quick training
echo ""
echo "Running quick training test..."
echo "----------------------------------------"

python -m src.train \
    --lr_dir "$LR_DIR" \
    --hr_dir "$HR_DIR" \
    --model "unet_sr" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --sample_limit "$SAMPLES" \
    --output_dir "$OUTPUT_DIR"

# Analyze results
echo ""
echo "========================================"
echo "Analysis Results"
echo "========================================"

python -c "
import torch
import sys
sys.path.insert(0, 'src')
from models import UNetSR

# Model test
model = UNetSR(in_channels=7, out_channels=2, scale_factor=2)
x = torch.randn(1, 7, 50, 51)
y = model(x)
print(f'Input shape: {x.shape}')
print(f'Output shape: {y.shape}')
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
print(f'Scale factor achieved: {y.shape[2] // x.shape[2]}x')
"

echo ""
echo "Test completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
