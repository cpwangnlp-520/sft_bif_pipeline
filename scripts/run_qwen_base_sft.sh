#!/bin/bash
set -euo pipefail
cd /workspace/new-preject/train-pipeline

export SWANLAB_PROJECT=Qwen2.5-Preliminary-test

CONFIG=configs/qwen_base_sft.yaml

echo "=== Qwen2.5-0.5B Base SFT (8 GPUs) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29500 \
    -m pipeline.cli train \
    --config "$CONFIG" \
    --run_name "qwen_base_sft_dolly" \
    --batch_size 16 \
    --grad_accum 1

echo "=== SFT DONE ==="
