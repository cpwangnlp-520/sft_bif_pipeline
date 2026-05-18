from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from .config import TrainConfig
from .data import ensure_chat_template, load_sft_dataset


def run_sft(
    config: TrainConfig,
    train_file_override: Optional[str] = None,
    run_name_override: Optional[str] = None,
    num_epochs_override: Optional[float] = None,
) -> str:
    train_file = train_file_override or config.train_file
    run_name = run_name_override or config.swanlab_run_sft_full
    num_epochs = num_epochs_override if num_epochs_override is not None else config.num_train_epochs
    output_dir = os.path.join(config.output_dir, run_name)

    print(f"[train] model={config.model_name_or_path}")
    print(f"[train] train_file={train_file}")
    print(f"[train] output_dir={output_dir}")
    print(f"[train] run_name={run_name}")
    print(f"[train] epochs={num_epochs}")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_chat_template(tokenizer, config.chat_template)

    print("[train] loading datasets...")
    train_ds = load_sft_dataset(train_file)
    eval_ds = {}
    for name, path in config.eval_files.items():
        eval_ds[name] = load_sft_dataset(path)
        print(f"  eval '{name}': {len(eval_ds[name])} samples")

    print("[train] loading model...")
    model_kwargs = {}
    if config.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)

    callbacks = []
    if config.use_swanlab:
        from swanlab.integration.transformers import SwanLabCallback

        swanlab_cb = SwanLabCallback(
            project=config.swanlab_project,
            experiment_name=run_name,
            config=config.to_dict(),
        )
        callbacks.append(swanlab_cb)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=config.eval_steps if eval_ds else None,
        seed=config.seed,
        max_length=config.cutoff_len,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=[],
        run_name=run_name,
        dataset_num_proc=config.preprocessing_num_workers,
        dataloader_num_workers=config.preprocessing_num_workers,
        save_total_limit=3,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if eval_ds else None,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    print("[train] starting training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"[train] model saved to {output_dir}")

    return output_dir
