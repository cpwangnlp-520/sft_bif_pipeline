#!/bin/bash
set -euo pipefail
cd /workspace/new-preject/train-pipeline

# ==============================================================================
# Phase 1: BIF sweep (2 tasks, GPU 0 and GPU 1 in parallel)
# ==============================================================================
echo "============================================"
echo "PHASE 1: BIF sweep (factqa + nq, parallel)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python -m bif.cli sweep-bif \
    --config runs/bif_1k_pool_factqa/bif_sweep/sweep.yaml &
BIF_PID1=$!

CUDA_VISIBLE_DEVICES=1 python -m bif.cli sweep-bif \
    --config runs/bif_1k_pool_nq/bif_sweep/sweep.yaml &
BIF_PID2=$!

echo "[bif] factqa PID=$BIF_PID1 (GPU 0)"
echo "[bif] nq     PID=$BIF_PID2 (GPU 1)"
wait $BIF_PID1 $BIF_PID2
echo "[bif] Both BIF sweeps done."

# ==============================================================================
# Phase 2: Prepare drop data
# ==============================================================================
echo ""
echo "============================================"
echo "PHASE 2: Prepare drop data"
echo "============================================"

find_bif_analysis() {
    local sweep_dir=$1
    for d in "${sweep_dir}/runs/"*/; do
        if [ -d "${d}analysis" ]; then
            echo "${d}analysis"
            return
        fi
    done
    echo ""
}

CODING_CSV=$(find_bif_analysis runs/bif_1k_pool_coding/bif_sweep)
FACTQA_CSV=$(find_bif_analysis runs/bif_1k_pool_factqa/bif_sweep)
NQ_CSV=$(find_bif_analysis runs/bif_1k_pool_nq/bif_sweep)

echo "[drop] coding  analysis: ${CODING_CSV:-NOT FOUND}"
echo "[drop] factqa  analysis: ${FACTQA_CSV:-NOT FOUND}"
echo "[drop] nq      analysis: ${NQ_CSV:-NOT FOUND}"

if [ -z "$CODING_CSV" ] || [ -z "$FACTQA_CSV" ] || [ -z "$NQ_CSV" ]; then
    echo "[drop] ERROR: Missing BIF analysis directory. Aborting."
    exit 1
fi

python -m pipeline.cli prepare-drop \
    --train_file data/gsm8k_sft_train.jsonl \
    --csv_dir "$CODING_CSV" \
    --strategies bottom top random \
    --k 200 \
    --pool_only --pool_size 1000 \
    --output_dir data/drop_coding

python -m pipeline.cli prepare-drop \
    --train_file data/gsm8k_sft_train.jsonl \
    --csv_dir "$FACTQA_CSV" \
    --strategies bottom top random \
    --k 200 \
    --pool_only --pool_size 1000 \
    --output_dir data/drop_factqa

python -m pipeline.cli prepare-drop \
    --train_file data/gsm8k_sft_train.jsonl \
    --csv_dir "$NQ_CSV" \
    --strategies bottom top random \
    --k 200 \
    --pool_only --pool_size 1000 \
    --output_dir data/drop_nq

# ==============================================================================
# Phase 3: SFT training (8 GPUs, serial)
# ==============================================================================
echo ""
echo "============================================"
echo "PHASE 3: SFT training (8 GPUs, serial)"
echo "============================================"

sft_train() {
    local config=$1
    local train_file=$2
    local run_name=$3
    echo ""
    echo "--- SFT: $run_name ---"
    torchrun --nproc_per_node=8 -m pipeline.cli train \
        --config "$config" \
        --train_file "$train_file" \
        --run_name "$run_name"
}

CONFIG_70M="configs/step10000.yaml"
CONFIG_160M="configs/160m_step10000.yaml"
CONFIG_410M="configs/410m_step10000.yaml"

# --- 70m: factqa + nq (coding already done) ---
echo ""
echo "=== 70m SFT: factqa drop ==="
sft_train $CONFIG_70M data/drop_factqa/gsm8k_sft_train_drop_bottom_200.jsonl 70m_factqa_drop_bottom200
sft_train $CONFIG_70M data/drop_factqa/gsm8k_sft_train_drop_top_200.jsonl     70m_factqa_drop_top200
sft_train $CONFIG_70M data/drop_factqa/gsm8k_sft_train_drop_random_200.jsonl  70m_factqa_drop_random200

echo ""
echo "=== 70m SFT: nq drop ==="
sft_train $CONFIG_70M data/drop_nq/gsm8k_sft_train_drop_bottom_200.jsonl 70m_nq_drop_bottom200
sft_train $CONFIG_70M data/drop_nq/gsm8k_sft_train_drop_top_200.jsonl     70m_nq_drop_top200
sft_train $CONFIG_70M data/drop_nq/gsm8k_sft_train_drop_random_200.jsonl  70m_nq_drop_random200

# --- 160m: coding + factqa + nq ---
echo ""
echo "=== 160m SFT: coding drop ==="
sft_train $CONFIG_160M data/drop_coding/gsm8k_sft_train_drop_bottom_200.jsonl 160m_coding_drop_bottom200
sft_train $CONFIG_160M data/drop_coding/gsm8k_sft_train_drop_top_200.jsonl     160m_coding_drop_top200
sft_train $CONFIG_160M data/drop_coding/gsm8k_sft_train_drop_random_200.jsonl  160m_coding_drop_random200

echo ""
echo "=== 160m SFT: factqa drop ==="
sft_train $CONFIG_160M data/drop_factqa/gsm8k_sft_train_drop_bottom_200.jsonl 160m_factqa_drop_bottom200
sft_train $CONFIG_160M data/drop_factqa/gsm8k_sft_train_drop_top_200.jsonl     160m_factqa_drop_top200
sft_train $CONFIG_160M data/drop_factqa/gsm8k_sft_train_drop_random_200.jsonl  160m_factqa_drop_random200

echo ""
echo "=== 160m SFT: nq drop ==="
sft_train $CONFIG_160M data/drop_nq/gsm8k_sft_train_drop_bottom_200.jsonl 160m_nq_drop_bottom200
sft_train $CONFIG_160M data/drop_nq/gsm8k_sft_train_drop_top_200.jsonl     160m_nq_drop_top200
sft_train $CONFIG_160M data/drop_nq/gsm8k_sft_train_drop_random_200.jsonl  160m_nq_drop_random200

# --- 410m: coding + factqa + nq ---
echo ""
echo "=== 410m SFT: coding drop ==="
sft_train $CONFIG_410M data/drop_coding/gsm8k_sft_train_drop_bottom_200.jsonl 410m_coding_drop_bottom200
sft_train $CONFIG_410M data/drop_coding/gsm8k_sft_train_drop_top_200.jsonl     410m_coding_drop_top200
sft_train $CONFIG_410M data/drop_coding/gsm8k_sft_train_drop_random_200.jsonl  410m_coding_drop_random200

echo ""
echo "=== 410m SFT: factqa drop ==="
sft_train $CONFIG_410M data/drop_factqa/gsm8k_sft_train_drop_bottom_200.jsonl 410m_factqa_drop_bottom200
sft_train $CONFIG_410M data/drop_factqa/gsm8k_sft_train_drop_top_200.jsonl     410m_factqa_drop_top200
sft_train $CONFIG_410M data/drop_factqa/gsm8k_sft_train_drop_random_200.jsonl  410m_factqa_drop_random200

echo ""
echo "=== 410m SFT: nq drop ==="
sft_train $CONFIG_410M data/drop_nq/gsm8k_sft_train_drop_bottom_200.jsonl 410m_nq_drop_bottom200
sft_train $CONFIG_410M data/drop_nq/gsm8k_sft_train_drop_top_200.jsonl     410m_nq_drop_top200
sft_train $CONFIG_410M data/drop_nq/gsm8k_sft_train_drop_random_200.jsonl  410m_nq_drop_random200

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
