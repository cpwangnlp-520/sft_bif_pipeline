# SFT-BIF Training Pipeline

A lightweight pipeline for SFT training with BIF (Bayesian Influence Function) data filtering.

## Pipeline Flow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Prepare Data │────>│  Round 1 SFT │────>│   BIF Sweep  │
│  (3500 sft)  │     │  (full data) │     │ (base vs     │
│  (1500 eval) │     │  (10 epochs) │     │  final ckpt) │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                          bottom_ids (lowest influence)
                                                 │
                                          ┌──────▼───────┐
                                          │   Filter     │
                                          │   bottom_k   │
                                          └──────┬───────┘
                                                 │
                                          ┌──────▼───────┐
                                          │  Round 2 SFT │
                                          │ (filtered,   │
                                          │ token-aligned│
                                          │  epochs)     │
                                          └──────────────┘
```

**SwanLab** tracks all experiments under one project with clear naming:

| Experiment | SwanLab Name |
|---|---|
| Round 1 SFT | `{experiment_name}_sft_full` |
| BIF sweep | `{experiment_name}_bif_sweep` |
| Round 2 SFT | `{experiment_name}_sft_filtered` |

## Output Structure

```
runs/{experiment_name}/
├── {experiment_name}_sft_full/       # Round 1 model
├── {experiment_name}_sft_filtered/   # Round 2 model
├── bif_sweep/                        # BIF sweep results
│   ├── base_run.yaml                 # Auto-generated base config
│   ├── sweep.yaml                    # Auto-generated sweep config
│   ├── traces/                       # BIF trace data
│   └── runs/                         # Per-grid-point results
├── bif_pool.jsonl                    # BIF pool data
└── bif_query.jsonl                   # BIF query data
```

## Install

```bash
pip install -e .

# Install BIF submodule
cd third_party/BIF && pip install -e . && cd ../..
```

## Data Preparation

```bash
python scripts/prepare_data.py --num_train 3500 --num_eval 500 --seed 42
```

Downloads and prepares:

| File | Source | Count | Purpose |
|---|---|---|---|
| `gsm8k_sft_train.jsonl` | GSM8K train | 3500 | SFT training |
| `gsm8k_val.jsonl` | GSM8K test | 660 | eval |
| `nq_eval.jsonl` | NaturalQuestions | 500 | eval + BIF query |
| `factqa_eval.jsonl` | TruthfulQA | 500 | eval + BIF query |
| `coding_eval.jsonl` | MBPP | 500 | eval + BIF query |

All eval datasets contain **free-form answers** (not multiple choice).

## Configuration

### SFT Config (`configs/sft_gsm8k.yaml`)

```yaml
experiment_name: gsm8k_pythia70m    # Used for output dirs & SwanLab naming
model_name_or_path: <YOUR_MODEL_PATH>
output_root: runs                    # All outputs under runs/{experiment_name}/

num_train_epochs: 10
train_file: data/gsm8k_sft_train.jsonl
eval_files:
  gsm8k: data/gsm8k_val.jsonl
  nq: data/nq_eval.jsonl
  factqa: data/factqa_eval.jsonl
  coding: data/coding_eval.jsonl

use_swanlab: true
swanlab_project: sft-bif-pipeline   # Fixed project for all experiments

bottom_k: 500                        # Number of bottom samples to remove
```

### BIF Config (`configs/bif.yaml`)

```yaml
num_chains: 2
draws_per_chain: 100
lr: 1.0e-4
gamma: 100.0
nbeta: 100.0
sampler_type: sgld

sweep_lr_values: [1.0e-4]
sweep_gamma_values: [100.0, 1000.0]
sweep_nbeta_values: [100.0]
```

All BIF parameters are configurable — no hardcoded values.

## Usage

### One-Click Pipeline

```bash
bash scripts/run_pipeline.sh configs/sft_gsm8k.yaml configs/bif.yaml 8
```

### Step by Step

```bash
# 1. SFT on full data
torchrun --nproc_per_node=8 -m pipeline.cli train --config configs/sft_gsm8k.yaml

# 2. Prepare BIF data (auto-converts SFT format to BIF pool/query)
python -m pipeline.cli prepare-bif --config configs/sft_gsm8k.yaml

# 3. Full pipeline (SFT -> BIF sweep -> filter -> re-SFT)
torchrun --nproc_per_node=8 -m pipeline.cli pipeline \
    --config configs/sft_gsm8k.yaml \
    --bif_config configs/bif.yaml \
    --num_gpus 8
```

### Filter Only (if BIF already ran)

```bash
python -m pipeline.cli filter \
    --train_file data/gsm8k_sft_train.jsonl \
    --bif_dir runs/gsm8k_pythia70m/bif_sweep/runs/grid_0000.../analysis \
    --checkpoint final_model \
    --bottom_k 500 \
    --output data/gsm8k_sft_train_filtered.jsonl
```

## Token Alignment

When bottom samples are removed, the pipeline automatically increases epochs so total training tokens stay constant:

```
Round 1: 3500 samples × 10.00 epochs = 35000 sample-epochs
Round 2: 3000 samples × 11.67 epochs = 35000 sample-epochs
```

## BIF Data Format

Both pool and query are **full Q+A with `answer_start_char`** — BIF only computes loss on answer tokens, matching SFT training behavior:

```
Input:  [question tokens] [answer tokens]
Loss:    IGNORE            ✓ computed
                           ↑ answer_start_char points here
```

## Update BIF

```bash
git submodule update --remote third_party/BIF
```
