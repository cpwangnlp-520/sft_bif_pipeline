from __future__ import annotations

import argparse
import json
import os

from .bif import find_bif_sweep_best_run, generate_bif_configs
from .config import BIFConfig, TrainConfig
from .data import (
    calc_aligned_epochs,
    filter_bif_bottom,
    read_bif_bottom_ids,
    sft_to_bif_pool,
    sft_to_bif_query,
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
    train_config = TrainConfig.from_yaml(args.config)
    bif_config = BIFConfig.from_yaml(args.bif_config) if args.bif_config else BIFConfig()

    pool_path = os.path.join(train_config.output_dir, "bif_pool.jsonl")
    sft_to_bif_pool(train_config.train_file, pool_path, source="sft")

    query_records = []
    for name, path in train_config.eval_files.items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                query_records.append(line)

    tmp_path = pool_path + ".query_tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(query_records)

    query_path = os.path.join(train_config.output_dir, "bif_query.jsonl")
    sft_to_bif_query(tmp_path, query_path, source="eval")
    os.remove(tmp_path)

    print(f"\n[prepare-bif] pool_jsonl:  {pool_path}")
    print(f"[prepare-bif] query_jsonl: {query_path}")


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
    sft_output = run_sft(train_config, run_name_override=train_config.swanlab_run_sft_full)

    # ========================================
    # Prepare BIF data
    # ========================================
    print("\n" + "=" * 60)
    print("PREPARING BIF data")
    print("=" * 60)

    pool_path = os.path.join(train_config.output_dir, "bif_pool.jsonl")
    sft_to_bif_pool(train_config.train_file, pool_path, source="sft")

    query_records = []
    for name, path in train_config.eval_files.items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                query_records.append(line)

    tmp_path = pool_path + ".query_tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(query_records)

    query_path = os.path.join(train_config.output_dir, "bif_query.jsonl")
    sft_to_bif_query(tmp_path, query_path, source="eval")
    os.remove(tmp_path)

    # ========================================
    # BIF sweep
    # ========================================
    print("\n" + "=" * 60)
    print("BIF SWEEP: base model vs final checkpoint")
    print("=" * 60)

    sweep_config_path = generate_bif_configs(
        train_config, bif_config, sft_output, pool_path, query_path
    )

    cmd = (
        f"SWANLAB_PROJECT={train_config.swanlab_project} "
        f"torchrun --standalone --nnodes=1 --nproc-per-node={args.num_gpus} "
        f"-m bif.cli sweep-bif --config {sweep_config_path}"
    )
    print(f"\n[bif] Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"[bif] sweep failed (code={ret}). Stopping.")
        return

    # ========================================
    # Read BIF results & filter bottom
    # ========================================
    print("\n" + "=" * 60)
    print("FILTERING bottom data")
    print("=" * 60)

    bif_dir = os.path.join(train_config.output_dir, "bif_sweep")
    best_run_dir = find_bif_sweep_best_run(bif_dir)
    analysis_dir = os.path.join(best_run_dir, "analysis")
    bottom_ids = read_bif_bottom_ids(analysis_dir, "final_model", train_config.bottom_k)

    filtered_path = train_config.train_file.replace(".jsonl", "_filtered.jsonl")
    stats = filter_bif_bottom(train_config.train_file, bottom_ids, filtered_path)

    # ========================================
    # Round 2: SFT on filtered data (token-aligned)
    # ========================================
    print("\n" + "=" * 60)
    print("ROUND 2: SFT on filtered data (token-aligned)")
    print("=" * 60)

    aligned_epochs = calc_aligned_epochs(original_epochs, original_count, stats["kept"])

    run_sft(
        train_config,
        train_file_override=filtered_path,
        run_name_override=train_config.swanlab_run_sft_filtered,
        num_epochs_override=aligned_epochs,
    )

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Experiment:        {train_config.experiment_name}")
    print(f"  Round 1 (full):    {train_config.output_dir}/{train_config.swanlab_run_sft_full}")
    print(f"  Round 2 (filtered):{train_config.output_dir}/{train_config.swanlab_run_sft_filtered}")
    print(f"  BIF results:       {bif_dir}")
    print(f"  Epochs aligned:    {original_epochs} -> {aligned_epochs:.2f}")
    print(f"  Samples:           {original_count} -> {stats['kept']}")
    print(f"  Bottom removed:    {stats['removed']}")
    print(f"  SwanLab project:   {train_config.swanlab_project}")


def main():
    parser = argparse.ArgumentParser(
        description="SFT-BIF Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Run SFT training")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--train_file", default=None)
    p_train.add_argument("--run_name", default=None)

    p_filter = subparsers.add_parser("filter", help="Filter bottom data from BIF analysis")
    p_filter.add_argument("--train_file", required=True)
    p_filter.add_argument("--bif_dir", required=True)
    p_filter.add_argument("--checkpoint", default="final_model")
    p_filter.add_argument("--bottom_k", type=int, default=0)
    p_filter.add_argument("--output", required=True)

    p_prep = subparsers.add_parser("prepare-bif", help="Convert SFT data to BIF format")
    p_prep.add_argument("--config", required=True)
    p_prep.add_argument("--bif_config", default=None)

    p_pipe = subparsers.add_parser("pipeline", help="Full pipeline: SFT -> BIF -> filter -> re-SFT")
    p_pipe.add_argument("--config", required=True)
    p_pipe.add_argument("--bif_config", default=None)
    p_pipe.add_argument("--num_gpus", type=int, default=8)

    args = parser.parse_args()
    {
        "train": cmd_train,
        "filter": cmd_filter,
        "prepare-bif": cmd_prepare_bif,
        "pipeline": cmd_pipeline,
    }[args.command](args)


if __name__ == "__main__":
    main()
