#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/step10000.yaml}
NUM_GPUS=${2:-8}

echo "=== SFT training ==="
torchrun --nproc_per_node=$NUM_GPUS -m pipeline.cli train \
    --config "$CONFIG"

echo "=== Training done ==="
