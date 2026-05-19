#!/usr/bin/env python3
"""Generate BIF sweep configs for factqa and nq queries (70m model)."""
from __future__ import annotations

import os
import yaml

POOL = os.path.abspath("data/bif_pool_1k.jsonl")
MODEL = "/workspace/pku_percy/models/pythia-70m-step10000"
TOKENIZER = MODEL
OUTPUT_ROOT = os.path.abspath("runs")

queries = {
    "factqa": os.path.abspath("data/bif_query_factqa.jsonl"),
    "nq": os.path.abspath("data/bif_query_nq.jsonl"),
}

for qname, query_path in queries.items():
    out_dir = os.path.join(OUTPUT_ROOT, f"bif_1k_pool_{qname}", "bif_sweep")

    base_run = {
        "model_name_or_path": MODEL,
        "tokenizer_path": TOKENIZER,
        "pool_jsonl": POOL,
        "query_jsonl": query_path,
        "out_dir": os.path.abspath(os.path.join(out_dir, "traces")),
        "num_chains": 2,
        "draws_per_chain": 800,
        "max_length": 512,
        "train_batch_size": 32,
        "eval_batch_size": 256,
        "pool_eval_subset": 0,
        "lr": 1e-5,
        "gamma": 1000.0,
        "nbeta_mode": "devinterp",
        "nbeta": 100.0,
        "noise_level": 1.0,
        "num_burnin_steps": 100,
        "num_steps_bw_draws": 2,
        "sampler_type": "sgld",
        "seed": 42,
        "dtype": "bfloat16",
        "model_tag": f"pythia_70m_{qname}",
    }

    sweep = {
        "base_run_config": os.path.abspath(os.path.join(out_dir, "base_run.yaml")),
        "output_dir": os.path.abspath(out_dir),
        "run_overrides": {
            "draws_per_chain": 800,
            "num_chains": 2,
            "num_burnin_steps": 100,
            "num_steps_bw_draws": 2,
            "sampler_type": "sgld",
        },
        "sweep": {
            "prefix": f"pythia_70m_{qname}",
            "lr": {"values": [1e-5]},
            "gamma": {"values": [1000.0]},
            "nbeta": {"values": [100.0]},
        },
        "baseline": {
            "run_nbeta_zero": False,
            "compare_to_nbeta_zero": False,
            "include_if_in_grid": True,
        },
        "analysis": {
            "score_col": "cross_corr_mean_over_queries",
            "top_k": 200,
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
                "top_k": [200],
                "bottom_k": [200],
                "score_col": "cross_corr_mean_over_queries",
                "pass_threshold": 0.4,
                "seed": 42,
                "min_draws": 8,
            },
            "chain_stability": {
                "enabled": True,
                "top_k": [200],
                "bottom_k": [200],
                "score_col": "cross_corr_mean_over_queries",
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

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "base_run.yaml"), "w") as f:
        yaml.dump(base_run, f, default_flow_style=False)
    with open(os.path.join(out_dir, "sweep.yaml"), "w") as f:
        yaml.dump(sweep, f, default_flow_style=False)

    print(f"[config] {qname} -> {out_dir}/sweep.yaml")
