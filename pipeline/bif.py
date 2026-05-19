from __future__ import annotations

import os

import yaml

from .config import BIFConfig, TrainConfig


def _short_model_tag(model_name_or_path: str) -> str:
    base = os.path.basename(os.path.normpath(model_name_or_path))
    return base.replace("-", "_").replace(".", "")


def resolve_checkpoint_path(sft_output_dir: str, ckpt_name: str, base_model_path: str) -> str:
    if ckpt_name == "base_model":
        return base_model_path
    link = os.path.join(sft_output_dir, ckpt_name)
    if os.path.exists(link):
        if os.path.islink(link):
            target = os.readlink(link)
            return os.path.abspath(os.path.join(sft_output_dir, target))
        return os.path.abspath(link)
    candidate = os.path.join(sft_output_dir, "checkpoint-*")
    import glob
    matches = sorted(glob.glob(candidate))
    if matches:
        return os.path.abspath(matches[-1])
    raise FileNotFoundError(f"Checkpoint '{ckpt_name}' not found in {sft_output_dir}")


def generate_bif_configs(
    train_config: TrainConfig,
    bif_config: BIFConfig,
    sft_output_dir: str,
    pool_jsonl: str,
    query_jsonl: str,
) -> dict[str, str]:
    """Generate one BIF sweep config per checkpoint.

    Each checkpoint gets its own directory and sweep config.
    Returns dict mapping ckpt_name -> sweep_yaml_path.
    """
    bif_dir = os.path.join(train_config.output_dir, "bif_sweep")
    model_tag = _short_model_tag(train_config.model_name_or_path)

    configs = {}
    for ckpt_name in train_config.bif_checkpoints:
        model_path = resolve_checkpoint_path(sft_output_dir, ckpt_name, train_config.model_name_or_path)
        ckpt_dir = os.path.join(bif_dir, ckpt_name)
        base_run_path = os.path.join(ckpt_dir, "base_run.yaml")
        sweep_path = os.path.join(ckpt_dir, "sweep.yaml")

        tag = f"{model_tag}_{ckpt_name}"

        base_run = {
            "model_name_or_path": os.path.abspath(model_path),
            "tokenizer_path": train_config.model_name_or_path,
            "pool_jsonl": os.path.abspath(pool_jsonl),
            "query_jsonl": os.path.abspath(query_jsonl),
            "out_dir": os.path.abspath(os.path.join(ckpt_dir, "traces")),
            "num_chains": bif_config.num_chains,
            "draws_per_chain": bif_config.draws_per_chain,
            "max_length": bif_config.max_length,
            "train_batch_size": bif_config.train_batch_size,
            "eval_batch_size": bif_config.eval_batch_size,
            "pool_eval_subset": bif_config.pool_eval_subset,
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
            "model_tag": tag,
        }

        os.makedirs(ckpt_dir, exist_ok=True)
        with open(base_run_path, "w") as f:
            yaml.dump(base_run, f, default_flow_style=False)

        sweep_config = {
            "base_run_config": os.path.abspath(base_run_path),
            "output_dir": os.path.abspath(ckpt_dir),
            "run_overrides": {
                "draws_per_chain": bif_config.draws_per_chain,
                "num_chains": bif_config.num_chains,
                "num_burnin_steps": bif_config.num_burnin_steps,
                "num_steps_bw_draws": bif_config.num_steps_bw_draws,
                "sampler_type": bif_config.sampler_type,
            },
            "sweep": {
                "prefix": tag,
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
                "top_k": bif_config.top_k,
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
                    "top_k": [bif_config.top_k],
                    "bottom_k": [bif_config.top_k],
                    "score_col": bif_config.score_col,
                    "pass_threshold": 0.4,
                    "seed": 42,
                    "min_draws": 8,
                },
                "chain_stability": {
                    "enabled": True,
                    "top_k": [bif_config.top_k],
                    "bottom_k": [bif_config.top_k],
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

        configs[ckpt_name] = sweep_path
        print(f"[bif] {ckpt_name} sweep -> {sweep_path} (model={model_path})")

    return configs


def find_sweep_runs(sweep_dir: str) -> list[str]:
    """Find all completed run directories under a sweep."""
    runs_dir = os.path.join(sweep_dir, "runs")
    if not os.path.isdir(runs_dir):
        return []

    results = []
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if os.path.isdir(run_dir) and os.path.exists(os.path.join(run_dir, "analysis")):
            results.append(run_dir)

    return results


def find_best_sweep_run(sweep_dir: str) -> str:
    """Find the best sweep run (first completed run with analysis)."""
    runs = find_sweep_runs(sweep_dir)
    if runs:
        print(f"[bif] best sweep run: {os.path.basename(runs[0])}")
        return runs[0]

    runs_dir = os.path.join(sweep_dir, "runs")
    if os.path.isdir(runs_dir):
        dirs = sorted(os.listdir(runs_dir))
        if dirs:
            return os.path.join(runs_dir, dirs[0])

    raise FileNotFoundError(f"No sweep results found in {sweep_dir}")
