#!/bin/bash
set -uo pipefail
cd /workspace/new-preject/train-pipeline

export SWANLAB_PROJECT=Qwen2.5-Alignment-SFT

CONFIG=configs/qwen_align_sft.yaml

echo "=== Qwen2.5-0.5B Alignment SFT (8 GPUs) ==="
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 \
    -m pipeline.cli train \
    --config "$CONFIG" \
    --run_name "qwen_align_sft_600" \
    --batch_size 8 \
    --grad_accum 1

echo "=== Alignment SFT DONE ==="
