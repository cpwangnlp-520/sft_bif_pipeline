#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/sft_gsm8k.yaml}
BIF_CONFIG=${2:-configs/bif.yaml}
NUM_GPUS=${3:-8}

echo "============================================================"
echo "Step 1/3: Prepare datasets"
echo "============================================================"
python scripts/prepare_data.py --num_train 3500 --num_eval 500 --seed 42

echo ""
echo "============================================================"
echo "Step 2/3: SFT on full data (8 GPUs)"
echo "============================================================"
torchrun --nproc_per_node=$NUM_GPUS -m pipeline.cli train \
    --config "$CONFIG"

echo ""
echo "============================================================"
echo "Step 3/3: Full pipeline (BIF + drop + re-SFT)"
echo "============================================================"
python -m pipeline.cli pipeline \
    --config "$CONFIG" \
    --bif_config "$BIF_CONFIG" \
    --gpu 0 \
    --num_gpus $NUM_GPUS

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
