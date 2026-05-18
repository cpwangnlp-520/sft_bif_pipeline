from __future__ import annotations

import os

import yaml

from .config import BIFConfig


def generate_bif_sweep_config(bif_config: BIFConfig, output_path: str) -> str:
    """Generate BIF sweep-bif YAML config for base + final_model only."""
    base_run_config_path = os.path.join(os.path.dirname(output_path), "bif_base_run.yaml")

    base_run = {
        "model_root": bif_config.model_root,
        "base_model_path": bif_config.base_model_path,
        "tokenizer_path": bif_config.tokenizer_path or bif_config.base_model_path,
        "pool_jsonl": bif_config.pool_jsonl,
        "query_jsonl": bif_config.query_jsonl,
        "out_dir": os.path.join(bif_config.out_dir, "traces"),
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
    }

    os.makedirs(os.path.dirname(base_run_config_path) or ".", exist_ok=True)
    with open(base_run_config_path, "w") as f:
        yaml.dump(base_run, f, default_flow_style=False)

    bottom_k = bif_config.bottom_k if bif_config.bottom_k > 0 else 150

    sweep_config = {
        "base_run_config": base_run_config_path,
        "output_dir": bif_config.out_dir,
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
            "top_k": bottom_k,
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
                "top_k": [bottom_k],
                "score_col": bif_config.score_col,
                "pass_threshold": 0.4,
                "seed": 42,
                "min_draws": 8,
            },
            "chain_stability": {
                "enabled": True,
                "top_k": [bottom_k],
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

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(sweep_config, f, default_flow_style=False)

    print(f"[bif] sweep config -> {output_path}")
    print(f"[bif] base run config -> {base_run_config_path}")
    return output_path


def find_bif_sweep_best_run(sweep_dir: str, score_col: str = "cross_corr_mean_over_queries") -> str:
    """Find the best sweep run (most stable) from sweep_summary.csv."""
    import csv

    summary_path = os.path.join(sweep_dir, "sweep_summary.csv")
    if not os.path.exists(summary_path):
        dirs = [d for d in os.listdir(sweep_dir) if d.startswith("grid_")]
        if dirs:
            return os.path.join(sweep_dir, sorted(dirs)[0])
        raise FileNotFoundError(f"No sweep results found in {sweep_dir}")

    with open(summary_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    if not reader:
        raise ValueError(f"sweep_summary.csv is empty: {summary_path}")

    best_row = reader[0]
    run_id = best_row.get("run_id", best_row.get("grid_point", ""))
    best_dir = os.path.join(sweep_dir, "runs", run_id) if os.path.exists(os.path.join(sweep_dir, "runs", run_id)) else sweep_dir

    print(f"[bif] best sweep run: {run_id} -> {best_dir}")
    return best_dir
