#!/bin/bash
set -uo pipefail
cd /workspace/new-preject/train-pipeline

export SWANLAB_PROJECT=Qwen2.5-Preliminary-test
export PYTHONPATH="/workspace/new-preject/train-pipeline/third_party/BIF/src:${PYTHONPATH:-}"

SWEEP_DIR=runs/qwen_align/bif_sweeps/sweep_configs

# 27 configs, 8 GPUs per round
# Round 1: GPU 0-7 (8 configs)
# Round 2: GPU 0-7 (8 configs)
# Round 3: GPU 0-7 (8 configs)
# Round 4: GPU 0-2 (3 configs)

run_sweep() {
    local gpu=$1
    local config=$2
    echo "[BIF] Starting on GPU $gpu: $(basename $config .yaml)"
    CUDA_VISIBLE_DEVICES=$gpu python -m bif.cli sweep-bif --config "$config"
    echo "[BIF] Done on GPU $gpu: $(basename $config .yaml)"
}

# Get all sweep configs sorted
configs=($(ls $SWEEP_DIR/sweep_*.yaml | sort))
total=${#configs[@]}
echo "=== Total $total BIF sweep configs ==="

# Round 1: configs 0-7
echo ""
echo "=== Round 1: configs 1-8 (8 GPUs) ==="
for i in $(seq 0 7); do
    run_sweep $i "${configs[$i]}" &
done
wait
echo "=== Round 1 done ==="

# Round 2: configs 8-15
echo ""
echo "=== Round 2: configs 9-16 (8 GPUs) ==="
for i in $(seq 8 15); do
    gpu=$((i - 8))
    run_sweep $gpu "${configs[$i]}" &
done
wait
echo "=== Round 2 done ==="

# Round 3: configs 16-23
echo ""
echo "=== Round 3: configs 17-24 (8 GPUs) ==="
for i in $(seq 16 23); do
    gpu=$((i - 16))
    run_sweep $gpu "${configs[$i]}" &
done
wait
echo "=== Round 3 done ==="

# Round 4: configs 24-26 (3 GPUs)
echo ""
echo "=== Round 4: configs 25-27 (3 GPUs) ==="
for i in $(seq 24 26); do
    gpu=$((i - 24))
    run_sweep $gpu "${configs[$i]}" &
done
wait
echo "=== Round 4 done ==="

echo ""
echo "============================================"
echo "ALL 27 BIF SWEEPS DONE"
echo "============================================"
