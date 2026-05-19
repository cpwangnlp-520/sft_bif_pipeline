from __future__ import annotations

import argparse
import os
import subprocess

from .bif import find_best_sweep_run, generate_bif_configs
from .config import BIFConfig, DropConfig, TrainConfig
from .data import (
    prepare_bif_query,
    prepare_drop_data,
    read_bif_ids,
    sft_to_bif_pool,
)
from .train import run_sft


def _build_train_cmd(config_path: str, num_gpus: int, train_file: str | None = None,
                     run_name: str | None = None) -> str:
    parts = []
    if num_gpus > 1:
        parts.append(f"torchrun --nproc_per_node={num_gpus}")
    else:
        parts.append("python")
    parts.append("-m pipeline.cli train")
    parts.append(f"--config {config_path}")
    if train_file:
        parts.append(f"--train_file {train_file}")
    if run_name:
        parts.append(f"--run_name {run_name}")
    return " ".join(parts)


def cmd_train(args):
    config = TrainConfig.from_yaml(args.config)
    overrides = {}
    if args.train_file:
        overrides["train_file_override"] = args.train_file
    if args.run_name:
        overrides["run_name_override"] = args.run_name
    if args.swanlab_group:
        overrides["swanlab_group_override"] = args.swanlab_group
    run_sft(config, **overrides)


def cmd_prepare_bif(args):
    train_config = TrainConfig.from_yaml(args.config)

    pool_path = os.path.join(train_config.output_dir, "bif_pool.jsonl")
    sft_to_bif_pool(train_config.train_file, pool_path, source="sft")

    query_path = os.path.join(train_config.output_dir, "bif_query.jsonl")
    prepare_bif_query(train_config.eval_files, train_config.bif_query_exclude, query_path)

    print(f"\n[prepare-bif] pool_jsonl:  {pool_path}")
    print(f"[prepare-bif] query_jsonl: {query_path}")


def cmd_prepare_drop(args):
    if args.config:
        train_config = TrainConfig.from_yaml(args.config)
        drop_cfg = train_config.get_drop_config()
        if drop_cfg is None:
            raise ValueError("No 'drop' section found in config yaml")
        train_path = args.train_file or train_config.train_file
        output_dir = args.output_dir or os.path.join(train_config.output_dir, "drop_data")
    else:
        if not args.train_file or not args.csv_dir:
            raise ValueError("Either --config or --train_file + --csv_dir is required")
        train_path = args.train_file
        drop_cfg = DropConfig(
            strategies=args.strategies,
            k=args.k,
            csv_dir=args.csv_dir,
            csv_name=args.csv_name,
            random_seed=args.seed,
            pool_only=args.pool_only,
            pool_size=args.pool_size,
        )
        output_dir = args.output_dir or "data/drop"

    results = prepare_drop_data(train_path, drop_cfg, output_dir)

    print(f"\n[prepare-drop] Generated {len(results)} datasets:")
    for name, path in results.items():
        print(f"  {name}: {path}")


def cmd_pipeline(args):
    train_config = TrainConfig.from_yaml(args.config)
    bif_config = BIFConfig.from_yaml(args.bif_config) if args.bif_config else BIFConfig()
    exp_name = train_config.experiment_name or train_config.auto_name

    original_count = sum(1 for _ in open(train_config.train_file))

    num_gpus = args.num_gpus

    # ========================================
    # Step 1: SFT on full data
    # ========================================
    print("=" * 60)
    print(f"STEP 1: SFT on full data ({num_gpus} GPU{'s' if num_gpus > 1 else ''})")
    print("=" * 60)
    sft_run_name = f"{exp_name}_sft_full"
    train_cmd = _build_train_cmd(args.config, num_gpus, run_name=sft_run_name)
    print(f"[sft] Command: {train_cmd}")
    ret = subprocess.call(train_cmd, shell=True)
    if ret != 0:
        print(f"[sft] Training failed (code={ret}). Aborting pipeline.")
        return
    sft_output = train_config.output_dir

    # ========================================
    # Step 2: Prepare BIF data
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 2: Preparing BIF data")
    print("=" * 60)

    pool_path = os.path.join(train_config.output_dir, "bif_pool.jsonl")
    sft_to_bif_pool(train_config.train_file, pool_path, source="sft")

    query_path = os.path.join(train_config.output_dir, "bif_query.jsonl")
    prepare_bif_query(train_config.eval_files, train_config.bif_query_exclude, query_path)

    # ========================================
    # Step 3: BIF sweep (one per checkpoint, serial)
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 3: BIF sweep (one config per checkpoint)")
    print("=" * 60)

    sweep_configs = generate_bif_configs(
        train_config, bif_config, sft_output, pool_path, query_path
    )

    bif_num_gpus = args.bif_num_gpus
    bif_gpu_ids = args.bif_gpu_ids

    for ckpt_name, sweep_path in sweep_configs.items():
        print(f"\n[bif] Running sweep for checkpoint: {ckpt_name} "
              f"({bif_num_gpus} GPU{'s' if bif_num_gpus > 1 else ''})")
        if bif_num_gpus > 1:
            cmd = (
                f"CUDA_VISIBLE_DEVICES={bif_gpu_ids} "
                f"torchrun --standalone --nnodes=1 --nproc_per_node={bif_num_gpus} "
                f"-m bif.cli sweep-bif --config {sweep_path}"
            )
        else:
            cmd = (
                f"CUDA_VISIBLE_DEVICES={bif_gpu_ids or args.gpu} "
                f"python -m bif.cli sweep-bif --config {sweep_path}"
            )
        print(f"[bif] Command: {cmd}")
        ret = subprocess.call(cmd, shell=True)
        if ret != 0:
            print(f"[bif] Sweep for {ckpt_name} failed (code={ret}). Skipping.")
            continue
        print(f"[bif] Sweep for {ckpt_name} completed.")

    # ========================================
    # Step 4: Prepare drop data from BIF results
    # ========================================
    print("\n" + "=" * 60)
    print("STEP 4: Preparing drop data from BIF results")
    print("=" * 60)

    drop_cfg = train_config.get_drop_config()
    if drop_cfg is None:
        drop_cfg = DropConfig(
            strategies=["bottom", "top", "random"],
            k=bif_config.top_k,
            random_seed=train_config.seed,
        )

    if not drop_cfg.csv_dir:
        if not train_config.bif_checkpoints:
            raise ValueError("No bif_checkpoints specified and no drop.csv_dir provided")
        first_ckpt = train_config.bif_checkpoints[0]
        sweep_dir = os.path.join(train_config.output_dir, "bif_sweep", first_ckpt)
        best_run = find_best_sweep_run(sweep_dir)
        drop_cfg.csv_dir = os.path.join(best_run, "analysis")

    drop_output_dir = os.path.join(train_config.output_dir, "drop_data")
    drop_results = prepare_drop_data(train_config.train_file, drop_cfg, drop_output_dir)

    # ========================================
    # Step 5: SFT on each drop variant (serial)
    # ========================================
    train_results = {}
    for strategy_name, drop_path in drop_results.items():
        print("\n" + "=" * 60)
        print(f"STEP 5: SFT on {strategy_name}-dropped data ({drop_cfg.k} removed)")
        print("=" * 60)

        run_name = f"{exp_name}_drop_{strategy_name}_{drop_cfg.k}"
        drop_cmd = _build_train_cmd(args.config, num_gpus, train_file=drop_path, run_name=run_name)
        print(f"[sft] Command: {drop_cmd}")
        ret = subprocess.call(drop_cmd, shell=True)
        if ret != 0:
            print(f"[sft] Training for {strategy_name} failed (code={ret}). Skipping.")
            continue
        train_results[strategy_name] = os.path.join(train_config.output_dir, run_name)

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Experiment:       {exp_name}")
    print(f"  Step 1 (full):    {train_config.output_dir}/{sft_run_name}")
    print(f"  BIF checkpoints:  {train_config.bif_checkpoints}")
    print(f"  BIF analysis:     {drop_cfg.csv_dir}")
    for name, path in train_results.items():
        print(f"  Step 5 ({name}):   {path}")
    print(f"  Original samples: {original_count}")
    print(f"  Dropped per run:  {drop_cfg.k}")
    print(f"  SwanLab project:  {train_config.swanlab_project}")


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
    p_train.add_argument("--swanlab_group", default=None, help="Override swanlab group for this run")

    p_prep = subparsers.add_parser("prepare-bif", help="Convert SFT data to BIF pool/query format")
    p_prep.add_argument("--config", required=True)

    p_drop = subparsers.add_parser("prepare-drop", help="Prepare drop datasets from BIF CSVs")
    p_drop.add_argument("--config", default=None)
    p_drop.add_argument("--train_file", default=None)
    p_drop.add_argument("--csv_dir", default=None)
    p_drop.add_argument("--csv_name", default="final_model")
    p_drop.add_argument("--strategies", nargs="+", default=["bottom", "top", "random"],
                        choices=["bottom", "top", "random"])
    p_drop.add_argument("--k", type=int, default=500)
    p_drop.add_argument("--seed", type=int, default=42)
    p_drop.add_argument("--pool_only", action="store_true")
    p_drop.add_argument("--pool_size", type=int, default=0)
    p_drop.add_argument("--output_dir", default=None)

    p_pipe = subparsers.add_parser("pipeline", help="Full pipeline: SFT -> BIF -> drop -> re-SFT")
    p_pipe.add_argument("--config", required=True)
    p_pipe.add_argument("--bif_config", default=None)
    p_pipe.add_argument("--gpu", default="0", help="GPU id for BIF sweep (single GPU, ignored if --bif_gpu_ids set)")
    p_pipe.add_argument("--num_gpus", type=int, default=1, help="GPU count for SFT training")
    p_pipe.add_argument("--bif_num_gpus", type=int, default=1, help="GPU count for BIF sweep (1=single-GPU, >1=torchrun)")
    p_pipe.add_argument("--bif_gpu_ids", default="", help="Comma-separated GPU IDs for BIF (e.g. '0,1,2,3'). Single GPU uses --gpu if not set")

    args = parser.parse_args()
    {
        "train": cmd_train,
        "prepare-bif": cmd_prepare_bif,
        "prepare-drop": cmd_prepare_drop,
        "pipeline": cmd_pipeline,
    }[args.command](args)


if __name__ == "__main__":
    main()
