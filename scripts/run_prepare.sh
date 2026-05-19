#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/step10000.yaml}
NUM_GPUS=${2:-1}

echo "=== Prepare BIF data ==="
python -m pipeline.cli prepare-bif --config "$CONFIG"

echo "=== Prepare drop data ==="
python -m pipeline.cli prepare-drop --config "$CONFIG"

echo "=== Done ==="
