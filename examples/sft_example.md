# SFT Training Example

## 1. Prepare Training Data

Training data must be in JSONL format with a `messages` field (OpenAI chat format):

```jsonl
{"id": "0", "messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"id": "1", "messages": [{"role": "user", "content": "Write a haiku about cats."}, {"role": "assistant", "content": "Soft paws on the floor,\nWhiskers twitch in morning light,\nPurring by the door."}]}
```

Alpaca format (`instruction`/`input`/`output`) is also auto-converted:

```jsonl
{"instruction": "Translate to French", "input": "Hello world", "output": "Bonjour le monde"}
```

Eval data uses the same format. You can have multiple eval files:

```jsonl
{"id": "eval_0", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

## 2. Create Config YAML

```yaml
# Model to fine-tune (HuggingFace model ID or local path)
model_name_or_path: Qwen/Qwen2.5-0.5B

# Training hyperparameters
num_train_epochs: 3
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
learning_rate: 2.0e-5
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
seed: 42

# Data
train_file: data/train.jsonl
eval_files:
  val: data/val.jsonl        # name "val" used as prefix in eval metrics
cutoff_len: 2048             # max sequence length (tokens)

# Logging & checkpoints
save_steps: 100
logging_steps: 10
eval_steps: 50

# Output
experiment_name: my_sft_experiment
output_root: runs

# SwanLab (optional, set use_swanlab: true to enable)
use_swanlab: true
swanlab_project: My-SFT-Project
swanlab_group: my_experiment_group
```

## 3. Run SFT Training

### Single GPU

```bash
python -m pipeline.cli train --config configs/my_sft.yaml
```

### Multi-GPU (torchrun)

```bash
torchrun --nproc_per_node=4 -m pipeline.cli train \
    --config configs/my_sft.yaml \
    --batch_size 4 \
    --grad_accum 2
```

### Override options

```bash
python -m pipeline.cli train \
    --config configs/my_sft.yaml \
    --run_name my_custom_run \
    --train_file data/other_train.jsonl \
    --swanlab_group other_group \
    --batch_size 16
```

## 4. Config Parameters Reference

### Required

| Parameter | Description |
|---|---|
| `model_name_or_path` | HuggingFace model ID or local path to the base model |
| `train_file` | Path to training JSONL file |

### Training

| Parameter | Default | Description |
|---|---|---|
| `num_train_epochs` | 10 | Number of training epochs |
| `max_steps` | -1 | Max steps (-1 = use epochs) |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 32 | Gradient accumulation steps |
| `learning_rate` | 5e-5 | Peak learning rate |
| `lr_scheduler_type` | cosine | LR scheduler (cosine, linear, constant, etc.) |
| `warmup_ratio` | 0.1 | Warmup ratio (0-1) |
| `bf16` | true | Use bfloat16 precision |
| `seed` | 42 | Random seed |
| `gradient_checkpointing` | false | Enable gradient checkpointing (saves VRAM, slower) |

### Data

| Parameter | Default | Description |
|---|---|---|
| `eval_files` | {} | Dict of {name: path} for eval datasets. Each gets separate loss tracking (e.g. `eval_val_loss`) |
| `cutoff_len` | 1024 | Max token length. Sequences longer than this are truncated |
| `preprocessing_num_workers` | 8 | Tokenization workers |

### Logging & Output

| Parameter | Default | Description |
|---|---|---|
| `save_steps` | 200 | Save checkpoint every N steps |
| `logging_steps` | 10 | Log metrics every N steps |
| `eval_steps` | 200 | Run evaluation every N steps |
| `experiment_name` | "experiment" | Experiment name (used as output subdirectory) |
| `output_root` | "runs" | Root output directory. Model saved to `{output_root}/{experiment_name}` |

### SwanLab

| Parameter | Default | Description |
|---|---|---|
| `use_swanlab` | false | Enable SwanLab experiment tracking |
| `swanlab_project` | "sft-bif-drop-compare" | SwanLab project name |
| `swanlab_group` | "" | SwanLab group (groups runs in same chart) |

### BIF (only used by `pipeline` command)

| Parameter | Default | Description |
|---|---|---|
| `bif_checkpoints` | ["final_model"] | Which checkpoints to run BIF on. "final_model" = the final saved model |
| `bif_query_exclude` | [] | Eval file names to exclude from BIF query pool |

## 5. Output Structure

```
runs/
  my_sft_experiment/
    model.safetensors          # final model
    config.json
    tokenizer.json
    checkpoint-100/            # intermediate checkpoints
    checkpoint-200/
    final_model -> .           # symlink to output dir
    training_args.bin
```
