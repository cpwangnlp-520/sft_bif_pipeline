from __future__ import annotations

import os

import yaml

from .config import BIFConfig, TrainConfig


def generate_bif_configs(
    train_config: TrainConfig,
    bif_config: BIFConfig,
    sft_output_dir: str,
    pool_jsonl: str,
    query_jsonl: str,
) -> str:
    """Generate BIF base-run + sweep YAML configs. Returns sweep config path."""
    bif_dir = os.path.join(train_config.output_dir, "bif_sweep")
    base_run_path = os.path.join(bif_dir, "base_run.yaml")
    sweep_path = os.path.join(bif_dir, "sweep.yaml")

    base_run = {
        "model_root": sft_output_dir,
        "base_model_path": train_config.model_name_or_path,
        "tokenizer_path": train_config.model_name_or_path,
        "pool_jsonl": pool_jsonl,
        "query_jsonl": query_jsonl,
        "out_dir": os.path.join(bif_dir, "traces"),
        "num_chains": bif_config.num_chains,
        "draws_per_chain": bif_config.draws_per_chain,
        "max_length": bif_config.max_length,
        "train_batch_size": bif_config.train_batch_size,
        "eval_batch_size": bif_config.eval_batch_size,
        "lr": bif_config.lr,
        "gamma": bif_config.gamma,
        "nbeta_mode": bif_config.nbeta_mode,
        "nbeta": bif_config.nbeta,
        "noise_level": bif_config.noise_level,
        "num_burnin_steps": bif_config.num_burnin_steps,
        "num_steps_bw_draws": bif_config.num_steps_bw_draws,
        "sampler_type": bif_config.sampler_type,
        "seed": bif_config.seed,
        "dtype": bif_config.dtype,
        "experiment_name": train_config.swanlab_run_bif,
    }

    os.makedirs(bif_dir, exist_ok=True)
    with open(base_run_path, "w") as f:
        yaml.dump(base_run, f, default_flow_style=False)

    sweep_config = {
        "base_run_config": base_run_path,
        "output_dir": bif_dir,
        "run_overrides": {
            "draws_per_chain": bif_config.draws_per_chain,
            "num_chains": bif_config.num_chains,
            "num_burnin_steps": bif_config.num_burnin_steps,
            "num_steps_bw_draws": bif_config.num_steps_bw_draws,
            "sampler_type": bif_config.sampler_type,
        },
        "sweep": {
            "lr": {"values": bif_config.sweep_lr_values},
            "gamma": {"values": bif_config.sweep_gamma_values},
            "nbeta": {"values": bif_config.sweep_nbeta_values},
        },
        "baseline": {
            "run_nbeta_zero": False,
            "compare_to_nbeta_zero": False,
            "include_if_in_grid": True,
        },
        "analysis": {
            "score_col": bif_config.score_col,
            "top_k": train_config.bottom_k,
            "enable_aux_query_plots": False,
            "negate_scores": False,
        },
        "diagnostics": {
            "checkpoint": None,
            "reduce_chains": "stack",
            "split_stability": {
                "enabled": True,
                "num_splits": 20,
                "split_fraction": 0.5,
                "top_k": [train_config.bottom_k],
                "score_col": bif_config.score_col,
                "pass_threshold": 0.4,
                "seed": 42,
                "min_draws": 8,
            },
            "chain_stability": {
                "enabled": True,
                "top_k": [train_config.bottom_k],
                "score_col": bif_config.score_col,
                "min_draws_per_chain": 4,
            },
        },
        "execution": {
            "run_bif": True,
            "analyze_bif": True,
            "diagnostics": True,
            "skip_existing": True,
            "continue_on_error": True,
            "dry_run": False,
            "extra_env": {},
        },
    }

    with open(sweep_path, "w") as f:
        yaml.dump(sweep_config, f, default_flow_style=False)

    print(f"[bif] base_run -> {base_run_path}")
    print(f"[bif] sweep    -> {sweep_path}")
    return sweep_path


def find_bif_sweep_best_run(sweep_dir: str) -> str:
    """Find the first sweep run directory."""
    import csv

    summary_path = os.path.join(sweep_dir, "sweep_summary.csv")
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        if reader:
            run_id = reader[0].get("run_id", reader[0].get("grid_point", ""))
            candidate = os.path.join(sweep_dir, "runs", run_id)
            if os.path.exists(candidate):
                print(f"[bif] best sweep run: {run_id}")
                return candidate

    runs_dir = os.path.join(sweep_dir, "runs")
    if os.path.isdir(runs_dir):
        dirs = sorted([d for d in os.listdir(runs_dir) if d.startswith("grid_")])
        if dirs:
            best = os.path.join(runs_dir, dirs[0])
            print(f"[bif] using sweep run: {dirs[0]}")
            return best

    raise FileNotFoundError(f"No sweep results found in {sweep_dir}")
