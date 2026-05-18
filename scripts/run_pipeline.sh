#!/bin/bash
# ==============================================================================
# SFT-BIF Pipeline
#
# Flow: SFT(full, N epochs) → BIF sweep (base vs final) → filter bottom → SFT(filtered, aligned epochs)
#
# Token alignment: filtered SFT gets more epochs so total tokens stay the same
# Eval split: half for BIF query, half for validation
# ==============================================================================
set -euo pipefail

CONFIG=${1:-configs/sft_gsm8k.yaml}
BIF_CONFIG=${2:-configs/bif.yaml}
NUM_GPUS=8

echo "============================================================"
echo "Step 1: Prepare data (run once)"
echo "============================================================"
python scripts/prepare_data.py --output_dir data

echo ""
echo "============================================================"
echo "Step 2: SFT on full data"
echo "============================================================"
torchrun --nproc_per_node=$NUM_GPUS -m pipeline.cli train \
    --config "$CONFIG" \
    --run_name sft_full

echo ""
echo "============================================================"
echo "Step 3: Prepare BIF data + split eval"
echo "============================================================"
python -m pipeline.cli prepare-bif \
    --train_config "$CONFIG" \
    --bif_config "$BIF_CONFIG" \
    --split_eval

echo ""
echo "============================================================"
echo "Step 4: BIF sweep (base model vs final checkpoint)"
echo "============================================================"
torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS \
    -m pipeline.cli sweep-bif \
    --bif_config "$BIF_CONFIG" \
    --run

echo ""
echo "============================================================"
echo "Step 5: Filter bottom data from BIF results"
echo "============================================================"
python -m pipeline.cli filter \
    --train_file data/gsm8k_sft_train.jsonl \
    --bif_dir saves/gsm8k_sft/bif_sweep/analysis \
    --checkpoint final_model \
    --output data/gsm8k_sft_train_filtered.jsonl

echo ""
echo "============================================================"
echo "Step 6: SFT on filtered data (token-aligned epochs)"
echo "============================================================"
torchrun --nproc_per_node=$NUM_GPUS -m pipeline.cli train \
    --config "$CONFIG" \
    --train_file data/gsm8k_sft_train_filtered.jsonl \
    --run_name sft_filtered

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
