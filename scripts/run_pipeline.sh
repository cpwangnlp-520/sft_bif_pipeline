#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/sft_gsm8k.yaml}
BIF_CONFIG=${2:-configs/bif.yaml}
NUM_GPUS=${3:-1}
BIF_NUM_GPUS=${4:-1}
BIF_GPU_IDS=${5:-}

echo "============================================================"
echo "Step 1/3: Prepare datasets"
echo "============================================================"
python scripts/prepare_data.py --num_train 3500 --num_eval 500 --seed 42

echo ""
echo "============================================================"
echo "Step 2/3: SFT on full data (${NUM_GPUS} GPU(s))"
echo "============================================================"
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$NUM_GPUS -m pipeline.cli train \
        --config "$CONFIG"
else
    python -m pipeline.cli train --config "$CONFIG"
fi

echo ""
echo "============================================================"
echo "Step 3/3: Full pipeline (BIF ${BIF_NUM_GPUS} GPU(s), SFT ${NUM_GPUS} GPU(s))"
echo "============================================================"
PIPE_ARGS="--config $CONFIG --bif_config $BIF_CONFIG --num_gpus $NUM_GPUS --bif_num_gpus $BIF_NUM_GPUS"
if [ -n "$BIF_GPU_IDS" ]; then
    PIPE_ARGS="$PIPE_ARGS --bif_gpu_ids $BIF_GPU_IDS"
fi
python -m pipeline.cli pipeline $PIPE_ARGS

echo ""
echo "============================================================"
echo "DONE"
echo "============================================================"
