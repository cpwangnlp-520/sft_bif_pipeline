# SFT-BIF Training Pipeline

Train SFT models, run BIF (Bayesian Influence Function) analysis to identify influential/harmful training samples, and compare BIF-guided data filtering against random filtering.

## Pipeline Flow

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Prepare Data  │────>│  SFT (full)  │────>│  BIF Sweep   │
│               │     │  1 or N GPUs │     │  1 GPU       │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                   │
                                           BIF scores per sample
                                           (top_k / bottom_k CSVs)
                                                   │
                     ┌─────────────────────────────┼─────────────────────────┐
                     │                             │                         │
               ┌─────▼──────┐              ┌───────▼─────┐          ┌───────▼─────┐
               │ drop bottom │              │  drop top   │          │ drop random │
               │ (harmful)   │              │(influential)│          │  (baseline) │
               └─────┬──────┘              └───────┬─────┘          └───────┬─────┘
                     │                             │                         │
               ┌─────▼──────┐              ┌───────▼─────┐          ┌───────▼─────┐
               │ Re-SFT     │              │  Re-SFT     │          │  Re-SFT     │
               └────────────┘              └─────────────┘          └─────────────┘
```

**Key insight**: If dropping bottom-BIF samples improves performance more than dropping random samples, BIF successfully identifies harmful training data.

## Install

```bash
pip install -e .
cd third_party/BIF && pip install -e . && cd ../..
```

## Data Preparation

```bash
python scripts/prepare_data.py --num_train 3500 --num_eval 500 --seed 42
```

| File | Source | Count | Purpose |
|---|---|---|---|
| `gsm8k_sft_train.jsonl` | GSM8K train | 3500 | SFT training + BIF pool |
| `gsm8k_val.jsonl` | GSM8K train | 500 | eval |
| `nq_eval.jsonl` | NaturalQuestions | 500 | eval + BIF query |
| `factqa_eval.jsonl` | TruthfulQA | 500 | eval + BIF query |
| `coding_eval.jsonl` | MBPP | 500 | eval + BIF query |

## Configuration

### Train Config (`configs/train.yaml`)

```yaml
model_name_or_path: /path/to/model
experiment_name: my_experiment

num_train_epochs: 10
per_device_train_batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 5.0e-5

train_file: data/train.jsonl
eval_files:
  gsm8k: data/gsm8k_val.jsonl
  nq: data/nq_eval.jsonl

use_swanlab: true
swanlab_project: sft-bif-pipeline
swanlab_group: ""            # Group experiments in SwanLab

bif_checkpoints: [final_model]   # Checkpoints to run BIF on
bif_query_exclude: [gsm8k]       # Exclude eval sets from BIF query

drop:
  strategies: [bottom, top, random]
  k: 500
  pool_only: false           # If true, random only samples from BIF pool
```

### BIF Config (`configs/bif.yaml`)

```yaml
num_chains: 2
draws_per_chain: 800
max_length: 512
train_batch_size: 32
eval_batch_size: 256
lr: 1.0e-5
gamma: 1000.0
nbeta: 100.0
sampler_type: sgld
top_k: 200                  # Number of top/bottom samples to identify
sweep_lr_values: [1.0e-5]
sweep_gamma_values: [1000.0]
sweep_nbeta_values: [100.0]
```

## Usage

### One-Click Pipeline

```bash
# Single GPU for everything
bash scripts/run_pipeline.sh configs/step10000.yaml configs/bif.yaml 1

# SFT 8 GPUs, BIF single GPU
bash scripts/run_pipeline.sh configs/step10000.yaml configs/bif.yaml 8 1 0

# SFT 8 GPUs, BIF 4 GPUs (GPU 0,1,2,3)
bash scripts/run_pipeline.sh configs/step10000.yaml configs/bif.yaml 8 4 0,1,2,3
```

### Step by Step

```bash
# 1. SFT on full data (single GPU)
python -m pipeline.cli train --config configs/step10000.yaml

# 1. SFT on full data (multi-GPU)
torchrun --nproc_per_node=8 -m pipeline.cli train --config configs/step10000.yaml

# 2. Prepare BIF pool/query data
python -m pipeline.cli prepare-bif --config configs/step10000.yaml

# 3. Full pipeline — single-GPU BIF
python -m pipeline.cli pipeline --config configs/step10000.yaml \
    --bif_config configs/bif.yaml --num_gpus 1

# 3. Full pipeline — multi-GPU BIF (4 GPUs, IDs 0-3)
python -m pipeline.cli pipeline --config configs/step10000.yaml \
    --bif_config configs/bif.yaml --bif_num_gpus 4 --bif_gpu_ids 0,1,2,3 --num_gpus 8

# 4. Prepare drop datasets from BIF results
python -m pipeline.cli prepare-drop --config configs/step10000.yaml

# 5. Re-train on each dropped variant
python -m pipeline.cli train \
    --config configs/step10000.yaml \
    --train_file runs/.../drop_data/gsm8k_sft_train_drop_bottom_500.jsonl \
    --run_name my_exp_drop_bottom_500
```

### prepare-drop Options

```bash
# From config (uses drop section)
python -m pipeline.cli prepare-drop --config configs/step10000.yaml

# Manual mode
python -m pipeline.cli prepare-drop \
    --train_file data/gsm8k_sft_train.jsonl \
    --csv_dir runs/.../bif_sweep/final_model/runs/basemodel_0000.../analysis \
    --strategies bottom top random \
    --k 200 \
    --pool_only --pool_size 1000    # Random only from first 1000 samples
```

## CLI Commands

| Command | Description |
|---|---|
| `train` | Run SFT training |
| `prepare-bif` | Convert SFT data to BIF pool/query format |
| `prepare-drop` | Prepare drop datasets from BIF CSVs |
| `pipeline` | Full pipeline: SFT -> BIF -> drop -> re-SFT |

## Output Structure

```
runs/{experiment_name}/
├── {name}_sft_full/              # Full-data SFT model
├── final_model -> .../           # Symlink for BIF
├── bif_pool.jsonl                # BIF pool (training data in BIF format)
├── bif_query.jsonl               # BIF query (eval data in BIF format)
├── bif_sweep/                    # BIF sweep results
│   ├── final_model/
│   │   ├── base_run.yaml
│   │   ├── sweep.yaml
│   │   ├── traces/
│   │   └── runs/
│   │       └── basemodel_0000_lr.../
│   │           └── analysis/final_model/
│   │               ├── pool_scores.csv
│   │               ├── top_200.csv
│   │               └── bottom_200.csv
│   └── base_model/              # (if bif_checkpoints includes base_model)
├── drop_data/                    # Prepared drop datasets
│   ├── gsm8k_sft_train_drop_bottom_500.jsonl
│   ├── gsm8k_sft_train_drop_top_500.jsonl
│   └── gsm8k_sft_train_drop_random_500.jsonl
└── {name}_drop_bottom_500/       # Re-trained models
├── {name}_drop_top_500/
└── {name}_drop_random_500/
```

## Important Notes

- **BIF sweep supports single-GPU and multi-GPU**. `--bif_num_gpus 1` uses `python -m` (default); `--bif_num_gpus N` uses `torchrun` to parallelize sweep grid points across N GPUs. Use `--bif_gpu_ids` to specify which GPUs (e.g. `0,1,2,3`).
- **SFT supports single-GPU (`python -m`) and multi-GPU (`torchrun`)**. The `pipeline` command uses `--num_gpus` to control this.
- **Don't change batch size** between experiments — steps must be comparable.
- **pool_only mode**: When BIF only analyzed a subset of training data (e.g., first 1000 of 3500), use `--pool_only --pool_size 1000` so random drop samples from the same pool for fair comparison.
- **SwanLab group**: Use `swanlab_group` to group related experiments (e.g., all drop-200 variants) in the same chart.
