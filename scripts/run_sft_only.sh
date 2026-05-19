#!/bin/bash
set -euo pipefail
cd /workspace/new-preject/train-pipeline

CONFIG_70M="configs/step10000.yaml"
CONFIG_160M="configs/160m_step10000.yaml"
CONFIG_410M="configs/410m_step10000.yaml"

sft_train() {
    local config=$1
    local train_file=$2
    local run_name=$3

    local model_dir
    model_dir=$(python -c "
from pipeline.config import TrainConfig
c = TrainConfig.from_yaml('$config')
print(c.output_dir)
")

    if [ -f "${model_dir}/${run_name}/trainer_state.json" ]; then
        echo "[skip] $run_name (already completed)"
        return 0
    fi

    echo ""
    echo "--- SFT: $run_name ---"
    torchrun --nproc_per_node=8 -m pipeline.cli train \
        --config "$config" \
        --train_file "$train_file" \
        --run_name "$run_name"
}

echo "============================================"
echo "SFT Training: sft-bif-drop-compare"
echo "============================================"

# --- 70m: factqa + nq (coding already done) ---
echo ""
echo "=== 70m: factqa drop ==="
sft_train $CONFIG_70M data/drop_factqa/gsm8k_sft_train_drop_bottom_200.jsonl 70m_factqa_drop_bottom200
sft_train $CONFIG_70M data/drop_factqa/gsm8k_sft_train_drop_top_200.jsonl     70m_factqa_drop_top200
sft_train $CONFIG_70M data/drop_factqa/gsm8k_sft_train_drop_random_200.jsonl  70m_factqa_drop_random200

echo ""
echo "=== 70m: nq drop ==="
sft_train $CONFIG_70M data/drop_nq/gsm8k_sft_train_drop_bottom_200.jsonl 70m_nq_drop_bottom200
sft_train $CONFIG_70M data/drop_nq/gsm8k_sft_train_drop_top_200.jsonl     70m_nq_drop_top200
sft_train $CONFIG_70M data/drop_nq/gsm8k_sft_train_drop_random_200.jsonl  70m_nq_drop_random200

# --- 160m: coding + factqa + nq ---
echo ""
echo "=== 160m: coding drop ==="
sft_train $CONFIG_160M data/drop_coding/gsm8k_sft_train_drop_bottom_200.jsonl 160m_coding_drop_bottom200
sft_train $CONFIG_160M data/drop_coding/gsm8k_sft_train_drop_top_200.jsonl     160m_coding_drop_top200
sft_train $CONFIG_160M data/drop_coding/gsm8k_sft_train_drop_random_200.jsonl  160m_coding_drop_random200

echo ""
echo "=== 160m: factqa drop ==="
sft_train $CONFIG_160M data/drop_factqa/gsm8k_sft_train_drop_bottom_200.jsonl 160m_factqa_drop_bottom200
sft_train $CONFIG_160M data/drop_factqa/gsm8k_sft_train_drop_top_200.jsonl     160m_factqa_drop_top200
sft_train $CONFIG_160M data/drop_factqa/gsm8k_sft_train_drop_random_200.jsonl  160m_factqa_drop_random200

echo ""
echo "=== 160m: nq drop ==="
sft_train $CONFIG_160M data/drop_nq/gsm8k_sft_train_drop_bottom_200.jsonl 160m_nq_drop_bottom200
sft_train $CONFIG_160M data/drop_nq/gsm8k_sft_train_drop_top_200.jsonl     160m_nq_drop_top200
sft_train $CONFIG_160M data/drop_nq/gsm8k_sft_train_drop_random_200.jsonl  160m_nq_drop_random200

# --- 410m: coding + factqa + nq ---
echo ""
echo "=== 410m: coding drop ==="
sft_train $CONFIG_410M data/drop_coding/gsm8k_sft_train_drop_bottom_200.jsonl 410m_coding_drop_bottom200
sft_train $CONFIG_410M data/drop_coding/gsm8k_sft_train_drop_top_200.jsonl     410m_coding_drop_top200
sft_train $CONFIG_410M data/drop_coding/gsm8k_sft_train_drop_random_200.jsonl  410m_coding_drop_random200

echo ""
echo "=== 410m: factqa drop ==="
sft_train $CONFIG_410M data/drop_factqa/gsm8k_sft_train_drop_bottom_200.jsonl 410m_factqa_drop_bottom200
sft_train $CONFIG_410M data/drop_factqa/gsm8k_sft_train_drop_top_200.jsonl     410m_factqa_drop_top200
sft_train $CONFIG_410M data/drop_factqa/gsm8k_sft_train_drop_random_200.jsonl  410m_factqa_drop_random200

echo ""
echo "=== 410m: nq drop ==="
sft_train $CONFIG_410M data/drop_nq/gsm8k_sft_train_drop_bottom_200.jsonl 410m_nq_drop_bottom200
sft_train $CONFIG_410M data/drop_nq/gsm8k_sft_train_drop_top_200.jsonl     410m_nq_drop_top200
sft_train $CONFIG_410M data/drop_nq/gsm8k_sft_train_drop_random_200.jsonl  410m_nq_drop_random200

echo ""
echo "============================================"
echo "ALL SFT DONE"
echo "============================================"
