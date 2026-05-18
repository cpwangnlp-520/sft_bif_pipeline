# SFT-BIF Training Pipeline

轻量级 SFT 训练 + BIF 数据筛选 pipeline。

## 架构

```
pipeline/
├── config.py    # 配置 (YAML → dataclass)
├── data.py      # 数据加载、格式转换、BIF 筛选
├── train.py     # SFT 训练 (trl.SFTTrainer + SwanLab)
└── cli.py       # CLI: train / filter / pipeline
```

## Pipeline 流程

```
SFT(全量数据) → BIF分析(外部) → 剔除bottom数据 → SFT(筛选后数据) → 对比结果
     ↓                              ↓
  model_v1                      model_v2
  eval: gsm8k/mmlu/factqa/coding  eval: gsm8k/mmlu/factqa/coding
```

## 安装

```bash
cd /workspace/new-preject/train-pipeline
pip install -e .
```

## 数据格式

训练和评估数据均为 JSONL，支持两种格式：

**格式 1 (推荐): messages 格式**
```jsonl
{"id": "gsm8k_0001", "messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}]}
```

**格式 2: alpaca 格式**
```jsonl
{"instruction": "问题", "input": "", "output": "答案"}
```

## BIF 结果格式

BIF 工具输出的 JSON 文件，包含需要剔除的 bottom 数据 ID：

```json
{
  "bottom_ids": ["gsm8k_0042", "gsm8k_0108", ...],
  "metadata": {
    "score_col": "cross_corr_mean_over_queries",
    "num_bottom": 500
  }
}
```

- `bottom_ids` 中的值对应训练数据中的 `id` 字段
- 如果训练数据没有 `id` 字段，则使用行索引（0-based 字符串）

## 使用方法

### 1. 准备数据

```bash
python scripts/prepare_data.py --output_dir data
```

下载 GSM8K 和 MMLU，生成标准 JSONL 格式。自定义数据集手动放入 `data/` 目录。

### 2. 单次 SFT 训练

```bash
# 单卡
python -m pipeline.cli train --config configs/sft_gsm8k.yaml

# 8卡
torchrun --nproc_per_node=8 -m pipeline.cli train --config configs/sft_gsm8k.yaml

# 覆盖参数
torchrun --nproc_per_node=8 -m pipeline.cli train \
    --config configs/sft_gsm8k.yaml \
    --run_name sft_full
```

训练会在每个 eval domain 上分别计算 loss：
- `eval_gsm8k_loss`
- `eval_mmlu_loss`
- `eval_factqa_loss`
- `eval_coding_loss`

### 3. BIF 筛选

```bash
python -m pipeline.cli filter \
    --train_file data/gsm8k_sft_train.jsonl \
    --bif_result bif_result.json \
    --output data/gsm8k_sft_train_filtered.jsonl
```

### 4. 重新训练

```bash
torchrun --nproc_per_node=8 -m pipeline.cli train \
    --config configs/sft_gsm8k.yaml \
    --train_file data/gsm8k_sft_train_filtered.jsonl \
    --run_name sft_filtered
```

### 5. 一键 Pipeline

```bash
bash scripts/run_pipeline.sh configs/sft_gsm8k.yaml bif_result.json
```

## 输出结构

```
saves/gsm8k_sft/
├── sft_full/              # Round 1: 全量数据训练
│   ├── checkpoint-200/
│   ├── checkpoint-400/
│   └── ...
└── sft_filtered/          # Round 2: BIF 筛选后训练
    ├── checkpoint-200/
    └── ...
```

## 配置说明

见 `configs/sft_gsm8k.yaml`，关键参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_name_or_path` | 模型路径 | - |
| `train_file` | 训练数据 JSONL | - |
| `eval_files` | 评估数据 dict (name: path) | {} |
| `cutoff_len` | 最大序列长度 | 1024 |
| `lora_rank` | LoRA rank, 0=全参训练 | 0 |
| `use_swanlab` | 是否使用 SwanLab | false |
| `swanlab_project` | SwanLab 项目名 | sft-bif-pipeline |
| `chat_template` | 自定义 chat template (可选) | null |

## 切换模型

只需修改 `model_name_or_path`，框架自动处理：
- Pythia: 无 chat template → 自动添加默认模板
- Llama/Qwen: 自带 chat template → 直接使用
- 大模型: 设置 `lora_rank: 8` + `gradient_checkpointing: true`
