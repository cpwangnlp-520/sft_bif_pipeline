"""Download and prepare eval datasets for SFT training."""
from __future__ import annotations

import json
import os
import argparse
import random


def prepare_gsm8k(output_dir: str = "data", num_train: int = 3500, num_val: int = 500, seed: int = 42) -> None:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    all_items = list(ds["train"])

    rng = random.Random(seed)
    rng.shuffle(all_items)

    train_items = all_items[:num_train]
    val_items = all_items[num_train:num_train + num_val]

    train_path = os.path.join(output_dir, "gsm8k_sft_train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(train_items):
            messages = [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ]
            f.write(json.dumps({"id": str(idx), "messages": messages}, ensure_ascii=False) + "\n")

    val_path = os.path.join(output_dir, "gsm8k_val.jsonl")
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_items:
            messages = [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"gsm8k: {len(train_items)} train, {len(val_items)} val (from train split, seed={seed})")


def prepare_nq(output_dir: str = "data", num_eval: int = 500, seed: int = 42) -> None:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    items = list(ds)

    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:num_eval]

    eval_path = os.path.join(output_dir, "nq_eval.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for item in items:
            answers = item["answer"]
            if isinstance(answers, str):
                answers = eval(answers)
            answer = answers[0] if answers else "unknown"

            messages = [
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": answer},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"nq (NaturalQuestions): {len(items)} eval (seed={seed})")


def prepare_coding(output_dir: str = "data", num_eval: int = 500, seed: int = 42) -> None:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", split="test")
    items = list(ds)

    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:num_eval]

    eval_path = os.path.join(output_dir, "coding_eval.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for item in items:
            prompt = item["text"]
            code = item["code"]
            messages = [
                {"role": "user", "content": f"Write a Python function: {prompt}"},
                {"role": "assistant", "content": code},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"coding (MBPP): {len(items)} eval (from {len(ds)} total, seed={seed})")


def prepare_factqa(output_dir: str = "data", num_eval: int = 500, seed: int = 42) -> None:
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    items = list(ds)

    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:num_eval]

    eval_path = os.path.join(output_dir, "factqa_eval.jsonl")
    with open(eval_path, "w", encoding="utf-8") as f:
        for item in items:
            question = item["question"]
            mc1 = item["mc1_targets"]
            choices = mc1["choices"]
            labels = mc1["labels"]
            correct_answer = choices[labels.index(1)] if 1 in labels else choices[0]

            options = "\n".join(f"- {c}" for c in choices)
            user_content = f"{question}\n\n{options}"
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": correct_answer},
            ]
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"factqa (TruthfulQA): {len(items)} eval (from {len(ds)} total, seed={seed})")


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT training and eval datasets")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "gsm8k", "nq", "coding", "factqa"])
    parser.add_argument("--num_train", type=int, default=3500, help="GSM8K train samples")
    parser.add_argument("--num_val", type=int, default=500, help="GSM8K val samples")
    parser.add_argument("--num_eval", type=int, default=500, help="Eval samples per domain")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    funcs = {
        "gsm8k": lambda: prepare_gsm8k(args.output_dir, args.num_train, args.num_val, args.seed),
        "nq": lambda: prepare_nq(args.output_dir, args.num_eval, args.seed),
        "coding": lambda: prepare_coding(args.output_dir, args.num_eval, args.seed),
        "factqa": lambda: prepare_factqa(args.output_dir, args.num_eval, args.seed),
    }

    if args.dataset == "all":
        for func in funcs.values():
            func()
    else:
        funcs[args.dataset]()

    print(f"\nDone! Data files in: {args.output_dir}")


if __name__ == "__main__":
    main()
