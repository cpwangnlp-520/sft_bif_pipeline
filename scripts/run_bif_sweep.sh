#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/step10000.yaml}
GPU=${2:-0}

echo "=== Running BIF sweep on GPU $GPU ==="
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
        echo "=== Sweeping: $(basename "$ckpt_dir") ==="
        CUDA_VISIBLE_DEVICES=$GPU python -m bif.cli sweep-bif --config "$SWEEP_YAML"
    fi
done

echo "=== BIF sweep done ==="
