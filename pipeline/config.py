from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import yaml


@dataclass
class TrainConfig:
    model_name_or_path: str = ""

    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 32
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    bf16: bool = True
    save_steps: int = 200
    logging_steps: int = 10
    eval_steps: int = 200
    seed: int = 42
    gradient_checkpointing: bool = False

    train_file: str = ""
    eval_files: dict[str, str] = field(default_factory=dict)
    cutoff_len: int = 1024
    preprocessing_num_workers: int = 8

    experiment_name: str = "experiment"
    output_root: str = "runs"

    use_swanlab: bool = False
    swanlab_project: str = "sft-bif-pipeline"

    chat_template: Optional[str] = None

    bottom_k: int = 500

    @classmethod
    def from_yaml(cls, path: str) -> TrainConfig:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def output_dir(self) -> str:
        return f"{self.output_root}/{self.experiment_name}"

    @property
    def swanlab_run_sft_full(self) -> str:
        return f"{self.experiment_name}_sft_full"

    @property
    def swanlab_run_sft_filtered(self) -> str:
        return f"{self.experiment_name}_sft_filtered"

    @property
    def swanlab_run_bif(self) -> str:
        return f"{self.experiment_name}_bif_sweep"


@dataclass
class BIFConfig:
    num_chains: int = 2
    draws_per_chain: int = 100
    max_length: int = 512
    train_batch_size: int = 32
    eval_batch_size: int = 64
    lr: float = 1e-4
    gamma: float = 100.0
    nbeta: float = 100.0
    nbeta_mode: str = "devinterp"
    noise_level: float = 1.0
    num_burnin_steps: int = 100
    num_steps_bw_draws: int = 2
    sampler_type: str = "sgld"
    seed: int = 42
    dtype: str = "bfloat16"

    score_col: str = "cross_corr_mean_over_queries"

    sweep_lr_values: list[float] = field(default_factory=lambda: [1e-4])
    sweep_gamma_values: list[float] = field(default_factory=lambda: [100.0, 1000.0])
    sweep_nbeta_values: list[float] = field(default_factory=lambda: [100.0])

    @classmethod
    def from_yaml(cls, path: str) -> BIFConfig:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
