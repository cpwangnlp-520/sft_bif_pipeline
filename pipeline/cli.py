from __future__ import annotations

import argparse
import json
import os

from .bif import find_bif_sweep_best_run, generate_bif_sweep_config
from .config import BIFConfig, TrainConfig
from .data import (
    calc_aligned_epochs,
    filter_bif_bottom,
    read_bif_bottom_ids,
    sft_to_bif_pool,
    sft_to_bif_query,
    split_eval_half,
)
from .train import run_sft


def cmd_train(args):
    config = TrainConfig.from_yaml(args.config)
    overrides = {}
    if args.train_file:
        overrides["train_file_override"] = args.train_file
    if args.run_name:
        overrides["run_name_override"] = args.run_name
    run_sft(config, **overrides)


def cmd_filter(args):
    bottom_ids = read_bif_bottom_ids(args.bif_dir, args.checkpoint, args.bottom_k)
    filter_bif_bottom(args.train_file, bottom_ids, args.output)


def cmd_prepare_bif(args):
    train_config = TrainConfig.from_yaml(args.train_config)
    bif_config = BIFConfig.from_yaml(args.bif_config) if args.bif_config else BIFConfig()

    pool_path = sft_to_bif_pool(train_config.train_file, args.pool_output)

    primary_eval = list(train_config.eval_files.values())[0]
    base = os.path.splitext(os.path.basename(primary_eval))[0]
    eval_dir = os.path.dirname(primary_eval)
    query_path = os.path.join(eval_dir, f"{base}_bif_query.jsonl")

    sft_to_bif_query(primary_eval, query_path)

    if args.split_eval:
        for name, path in train_config.eval_files.items():
            split_eval_half(
                path,
                path.replace(".jsonl", "_query.jsonl"),
                path.replace(".jsonl", "_val.jsonl"),
                seed=train_config.seed,
            )

    print(f"\n[prepare-bif] pool_jsonl: {pool_path}")
    print(f"[prepare-bif] query_jsonl: {query_path}")


def cmd_sweep_bif(args):
    bif_config = BIFConfig.from_yaml(args.bif_config)
    sweep_config_path = generate_bif_sweep_config(bif_config, args.sweep_config_output or "configs/bif_sweep.yaml")

    cmd = (
        f"torchrun --standalone --nnodes=1 --nproc-per-node={args.num_gpus} "
        f"-m bif.cli sweep-bif --config {sweep_config_path}"
    )
    print(f"\n[bif] Run BIF sweep with:")
    print(f"  {cmd}")
    print(f"\n[bif] Or run directly if BIF is installed:")
    os.system(cmd) if args.run else print(f"[bif] Dry run. Add --run to execute.")


def cmd_pipeline(args):
    train_config = TrainConfig.from_yaml(args.config)
    bif_config = BIFConfig.from_yaml(args.bif_config) if args.bif_config else BIFConfig()

    original_count = sum(1 for _ in open(train_config.train_file))
    original_epochs = train_config.num_train_epochs

    # ========================================
    # Round 1: SFT on full data
    # ========================================
    print("=" * 60)
    print("ROUND 1: SFT on full data")
    print("=" * 60)
    sft_output = run_sft(train_config, run_name_override="sft_full")

    # ========================================
    # Prepare BIF data
    # ========================================
    print("\n" + "=" * 60)
    print("PREPARING BIF data")
    print("=" * 60)

    pool_path = sft_to_bif_pool(train_config.train_file, "data/bif_pool.jsonl")

    primary_eval = list(train_config.eval_files.values())[0]
    query_path = "data/bif_query.jsonl"
    val_path = "data/bif_val.jsonl"
    split_eval_half(primary_eval, query_path, val_path, seed=train_config.seed)

    # ========================================
    # Run BIF sweep
    # ========================================
    print("\n" + "=" * 60)
    print("BIF SWEEP: base model vs final checkpoint")
    print("=" * 60)

    bif_config.pool_jsonl = pool_path
    bif_config.query_jsonl = query_path
    bif_config.model_root = sft_output
    bif_config.base_model_path = train_config.model_name_or_path
    bif_config.tokenizer_path = train_config.model_name_or_path
    if bif_config.out_dir == "runs/bif":
        bif_config.out_dir = os.path.join(train_config.output_dir, "bif_sweep")

    sweep_config_path = generate_bif_sweep_config(bif_config, "configs/bif_sweep_generated.yaml")

    cmd = (
        f"torchrun --standalone --nnodes=1 --nproc-per-node={args.num_gpus} "
        f"-m bif.cli sweep-bif --config {sweep_config_path}"
    )
    print(f"\n[bif] Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"[bif] sweep failed with code {ret}. Stopping pipeline.")
        return

    # ========================================
    # Read BIF results & filter bottom data
    # ========================================
    print("\n" + "=" * 60)
    print("FILTERING bottom data from BIF results")
    print("=" * 60)

    best_run_dir = find_bif_sweep_best_run(bif_config.out_dir, bif_config.score_col)
    bottom_ids = read_bif_bottom_ids(
        os.path.join(best_run_dir, "analysis"),
        checkpoint_name="final_model",
        bottom_k=bif_config.bottom_k,
    )

    filtered_path = train_config.train_file.replace(".jsonl", "_filtered.jsonl")
    stats = filter_bif_bottom(train_config.train_file, bottom_ids, filtered_path)

    # ========================================
    # Round 2: SFT on filtered data (token-aligned)
    # ========================================
    print("\n" + "=" * 60)
    print("ROUND 2: SFT on filtered data (token-aligned)")
    print("=" * 60)

    aligned_epochs = calc_aligned_epochs(original_epochs, original_count, stats["kept"])

    eval_files_aligned = {}
    for name, path in train_config.eval_files.items():
        val_file = path.replace(".jsonl", "_val.jsonl")
        eval_files_aligned[name] = val_file if os.path.exists(val_file) else path

    train_config.eval_files = eval_files_aligned

    run_sft(
        train_config,
        train_file_override=filtered_path,
        run_name_override="sft_filtered",
        num_epochs_override=aligned_epochs,
    )

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Round 1 (full):     {sft_output}")
    print(f"  Round 2 (filtered): {train_config.output_dir}/sft_filtered")
    print(f"  Epochs aligned:     {original_epochs} -> {aligned_epochs:.2f}")
    print(f"  Samples:            {original_count} -> {stats['kept']}")
    print(f"  BIF bottom removed: {stats['removed']}")
    print(f"  SwanLab project:    {train_config.swanlab_project}")


def main():
    parser = argparse.ArgumentParser(
        description="SFT-BIF Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single SFT run
  torchrun --nproc_per_node=8 -m pipeline.cli train --config configs/sft_gsm8k.yaml

  # Prepare BIF data (convert SFT format to BIF format)
  python -m pipeline.cli prepare-bif --train_config configs/sft_gsm8k.yaml --split_eval

  # Run BIF sweep separately
  torchrun --nproc_per_node=8 -m pipeline.cli sweep-bif --bif_config configs/bif.yaml

  # Filter bottom data from existing BIF analysis
  python -m pipeline.cli filter --train_file data/train.jsonl --bif_dir runs/bif/analysis --checkpoint final_model --output data/filtered.jsonl

  # Full pipeline: SFT -> BIF sweep -> filter -> re-SFT (token-aligned)
  torchrun --nproc_per_node=8 -m pipeline.cli pipeline --config configs/sft_gsm8k.yaml --bif_config configs/bif.yaml
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Run SFT training")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--train_file", default=None)
    p_train.add_argument("--run_name", default=None)

    # filter
    p_filter = subparsers.add_parser("filter", help="Filter bottom data from BIF analysis results")
    p_filter.add_argument("--train_file", required=True)
    p_filter.add_argument("--bif_dir", required=True, help="BIF analysis output directory")
    p_filter.add_argument("--checkpoint", default="final_model", help="Checkpoint name in BIF results")
    p_filter.add_argument("--bottom_k", type=int, default=0, help="Number of bottom samples (0=auto)")
    p_filter.add_argument("--output", required=True)

    # prepare-bif
    p_prep = subparsers.add_parser("prepare-bif", help="Convert SFT data to BIF pool/query format")
    p_prep.add_argument("--train_config", required=True, help="SFT training YAML config")
    p_prep.add_argument("--bif_config", default=None, help="BIF config YAML (optional)")
    p_prep.add_argument("--pool_output", default="data/bif_pool.jsonl")
    p_prep.add_argument("--split_eval", action="store_true", help="Split eval in half: query + val")

    # sweep-bif
    p_sweep = subparsers.add_parser("sweep-bif", help="Generate and run BIF sweep")
    p_sweep.add_argument("--bif_config", required=True, help="BIF config YAML")
    p_sweep.add_argument("--sweep_config_output", default=None)
    p_sweep.add_argument("--num_gpus", type=int, default=8)
    p_sweep.add_argument("--run", action="store_true", help="Actually run (default: dry-run)")

    # pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Full pipeline: SFT -> BIF -> filter -> re-SFT")
    p_pipe.add_argument("--config", required=True, help="SFT training YAML config")
    p_pipe.add_argument("--bif_config", default=None, help="BIF config YAML (optional)")
    p_pipe.add_argument("--num_gpus", type=int, default=8)

    args = parser.parse_args()
    {
        "train": cmd_train,
        "filter": cmd_filter,
        "prepare-bif": cmd_prepare_bif,
        "sweep-bif": cmd_sweep_bif,
        "pipeline": cmd_pipeline,
    }[args.command](args)


if __name__ == "__main__":
    main()
