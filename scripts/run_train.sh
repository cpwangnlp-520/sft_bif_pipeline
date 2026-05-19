#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/step10000.yaml}
NUM_GPUS=${2:-1}

echo "=== SFT training (${NUM_GPUS} GPU(s)) ==="
if [ "$NUM_GPUS" -gt 1 ]; then
    torchrun --nproc_per_node=$NUM_GPUS -m pipeline.cli train \
        --config "$CONFIG"
else
    python -m pipeline.cli train --config "$CONFIG"
fi

echo "=== Training done ==="
