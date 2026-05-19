#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/step10000.yaml}
BIF_NUM_GPUS=${2:-1}
BIF_GPU_IDS=${3:-0}

echo "=== Prepare BIF data ==="
python -m pipeline.cli prepare-bif --config "$CONFIG"

BIF_DIR=$(python -c "
from pipeline.config import TrainConfig
c = TrainConfig.from_yaml('$CONFIG')
print(c.output_dir)
")

BIF_SWEEP_DIR="${BIF_DIR}/bif_sweep"

for ckpt_dir in "${BIF_SWEEP_DIR}"/*/; do
    SWEEP_YAML="${ckpt_dir}sweep.yaml"
    if [ -f "$SWEEP_YAML" ]; then
        echo "=== Sweeping: $(basename "$ckpt_dir") (${BIF_NUM_GPUS} GPU(s)) ==="
        if [ "$BIF_NUM_GPUS" -gt 1 ]; then
            CUDA_VISIBLE_DEVICES=$BIF_GPU_IDS \
                torchrun --standalone --nnodes=1 --nproc_per_node=$BIF_NUM_GPUS \
                -m bif.cli sweep-bif --config "$SWEEP_YAML"
        else
            CUDA_VISIBLE_DEVICES=$BIF_GPU_IDS \
                python -m bif.cli sweep-bif --config "$SWEEP_YAML"
        fi
    fi
done

echo "=== BIF sweep done ==="
