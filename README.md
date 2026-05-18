# SFT-BIF Training Pipeline

A lightweight pipeline for SFT training with BIF (Bayesian Influence Function) data filtering.

## Pipeline Flow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Prepare Data │────>│  Round 1 SFT │────>│   BIF Sweep  │
│  (3500 sft)  │     │  (full data) │     │ (base vs     │
│  (2000 eval) │     │  (10 epochs) │     │  final ckpt) │
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
| BIF trace | `bif_{run_id}` |
| BIF analysis | `analyze_{run_id}` |
| BIF diagnostics | `diagnostics_{run_id}` |
| Round 2 SFT | `{experiment_name}_sft_filtered` |

## BIF Sweep Flow

BIF sweep runs a grid search over SGLD hyperparameters (lr × gamma × nbeta). For each grid point:

```
┌─────────────────────────────────────────────────────────────┐
│                    BIF Sweep (per grid point)                 │
│                                                              │
│  1. run-bif                                                  │
│     ├─ Load base_model + final_model checkpoints             │
│     ├─ Run SGLD sampler on each checkpoint                   │
│     │   (num_chains × draws_per_chain draws per checkpoint)  │
│     └─ Save loss traces: observable_loss_trace.npz           │
│                    query_loss_trace.npz                       │
│                    step_loss_trace.npz                        │
│                                                              │
│  2. analyze-bif                                              │
│     ├─ Compute BIF scores per pool sample                    │
│     │   score = cross_corr_mean_over_queries                 │
│     ├─ Rank all pool samples by influence score              │
│     ├─ Save: pool_scores.csv, top_K.csv, bottom_K.csv        │
│     └─ Generate correlation heatmaps & distribution plots    │
│                                                              │
│  3. diagnostics                                               │
│     ├─ Split stability: split pool K times, check top-K      │
│     │   overlap (Spearman correlation between splits)         │
│     ├─ Chain stability: compare scores across SGLD chains    │
│     └─ Save: diagnostics_summary.csv, stability charts       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Multi-GPU**: Each GPU runs a disjoint subset of grid points in parallel. BIF traces
within each grid point are single-process (one GPU per point). With 8 GPUs and 18 grid
points, each GPU handles ~3 points sequentially.

**Output per grid point:**

```
bif_sweep/runs/grid_0000_lr0p0001_gamma100_nbeta100/
├── traces/                    # run-bif output
│   ├── base_model/            # BIF traces for base model
│   │   ├── chain_000/
│   │   └── chain_001/
│   └── final_model/           # BIF traces for final model
│       ├── chain_000/
│       └── chain_001/
├── analysis/                  # analyze-bif output
│   ├── base_model/
│   │   └── pool_scores.csv
│   └── final_model/
│       ├── pool_scores.csv    # All samples with influence scores
│       ├── top_500.csv        # Most influential samples
│       └── bottom_500.csv     # Least influential (harmful) samples
├── diagnostics/               # stability diagnostics
│   └── final_model/
│       └── diagnostics_summary.csv
├── run_config.yaml
├── analysis_config.yaml
└── sweep_run_manifest.json
```

**Sweep summary** (`bif_sweep/sweep_summary.csv`): one row per grid point with
lr, gamma, nbeta, status, elapsed time, and diagnostic metrics.

## Output Structure

```
runs/{experiment_name}/
├── {experiment_name}_sft_full/       # Round 1 model
├── final_model -> {experiment_name}_sft_full/  # Symlink for BIF
├── {experiment_name}_sft_filtered/   # Round 2 model
├── bif_sweep/                        # BIF sweep results
│   ├── base_run.yaml                 # Auto-generated base config
│   ├── sweep.yaml                    # Auto-generated sweep config
│   ├── sweep_plan.csv                # Grid plan (lr × gamma × nbeta)
│   ├── sweep_summary.csv             # Results summary
│   ├── traces/                       # BIF trace data
│   └── runs/                         # Per-grid-point results
│       ├── grid_0000_.../
│       ├── grid_0001_.../
│       └── ...
├── bif_pool.jsonl                    # BIF pool data (full Q+A)
└── bif_query.jsonl                   # BIF query data (full Q+A)
```

## Install

```bash
pip install -e .

# Install BIF submodule
cd third_party/BIF && pip install -e . && cd ../..
```

## Data Preparation

```bash
python scripts/prepare_data.py --num_train 3500 --num_val 500 --num_eval 500 --seed 42
```

Downloads and prepares:

| File | Source | Count | Purpose |
|---|---|---|---|
| `gsm8k_sft_train.jsonl` | GSM8K train | 3500 | SFT training |
| `gsm8k_val.jsonl` | GSM8K train | 500 | eval |
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
gradient_accumulation_steps: 1       # 1 step = batch_size × 1 × num_gpus
cutoff_len: 2048                     # Max token length
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
# SGLD sampler parameters
num_chains: 2
draws_per_chain: 100
num_burnin_steps: 100
num_steps_bw_draws: 2
sampler_type: sgld
max_length: 512
train_batch_size: 32
eval_batch_size: 128

# Default sweep grid point
lr: 1.0e-4
gamma: 100.0
nbeta: 100.0

# Sweep grid (2 × 3 × 3 = 18 points)
sweep_lr_values: [1.0e-4, 1.0e-5]
sweep_gamma_values: [10.0, 100.0, 1000.0]
sweep_nbeta_values: [10.0, 100.0, 1000.0]
```

All parameters are configurable — no hardcoded values.

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
