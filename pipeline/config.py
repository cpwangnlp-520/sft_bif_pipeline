from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import yaml


def _short_model_name(path: str) -> str:
    base = os.path.basename(os.path.normpath(path))
    return base.replace("-", "_").replace(".", "")


@dataclass
class DropConfig:
    strategies: list[str] = field(default_factory=lambda: ["bottom", "top", "random"])
    k: int = 500
    csv_dir: str = ""
    csv_name: str = "final_model"
    random_seed: int = 42
    pool_only: bool = False
    pool_size: int = 0

    VALID = {"bottom", "top", "random"}

    def __post_init__(self):
        bad = set(self.strategies) - self.VALID
        if bad:
            raise ValueError(f"Invalid drop strategies: {bad}")

    @classmethod
    def from_dict(cls, d: dict) -> DropConfig:
        keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass
class TrainConfig:
    model_name_or_path: str = ""

    num_train_epochs: int = 10
    max_steps: int = -1
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
    swanlab_group: str = ""

    chat_template: Optional[str] = None

    bif_checkpoints: list[str] = field(default_factory=lambda: ["final_model"])
    bif_query_exclude: list[str] = field(default_factory=list)

    drop: Optional[dict] = None

    @classmethod
    def from_yaml(cls, path: str) -> TrainConfig:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in keys})

    def to_dict(self) -> dict:
        return asdict(self)

    def get_drop_config(self) -> Optional[DropConfig]:
        if self.drop is None:
            return None
        return DropConfig.from_dict(self.drop)

    @property
    def auto_name(self) -> str:
        model = _short_model_name(self.model_name_or_path)
        lr_val = self.learning_rate
        if lr_val >= 1:
            lr_str = f"{lr_val:.0f}"
        else:
            exp = 0
            v = lr_val
            while v < 1:
                v *= 10
                exp += 1
            lr_str = f"{v:.0f}em{exp:02d}"
        return f"{model}_ep{self.num_train_epochs}_bs{self.per_device_train_batch_size}_lr{lr_str}"

    @property
    def output_dir(self) -> str:
        name = self.experiment_name or self.auto_name
        return f"{self.output_root}/{name}"


@dataclass
class BIFConfig:
    num_chains: int = 2
    draws_per_chain: int = 100
    max_length: int = 1024
    train_batch_size: int = 32
    eval_batch_size: int = 128
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
    pool_eval_subset: int = 0
    sweep_lr_values: list[float] = field(default_factory=lambda: [1e-4])
    sweep_gamma_values: list[float] = field(default_factory=lambda: [100.0, 1000.0])
    sweep_nbeta_values: list[float] = field(default_factory=lambda: [100.0])
    top_k: int = 500

    @classmethod
    def from_yaml(cls, path: str) -> BIFConfig:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        keys = set(cls.__dataclass_fields__.keys())
        return cls(**{k: v for k, v in d.items() if k in keys})
