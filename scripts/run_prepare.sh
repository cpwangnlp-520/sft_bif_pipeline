#!/bin/bash
set -euo pipefail

CONFIG=${1:-configs/step10000.yaml}
GPU=${2:-0}

echo "=== Prepare BIF data ==="
python -m pipeline.cli prepare-bif --config "$CONFIG"

echo "=== Prepare drop data ==="
python -m pipeline.cli prepare-drop --config "$CONFIG"

echo "=== Done ==="
