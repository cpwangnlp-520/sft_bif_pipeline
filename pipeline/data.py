from __future__ import annotations

import csv
import json
import math
import os
import random
from typing import Optional

from datasets import load_dataset


DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "### User:\n{{ message['content'] }}\n\n"
    "{% elif message['role'] == 'assistant' %}"
    "### Assistant:\n{{ message['content'] }}\n\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "### Assistant:\n"
    "{% endif %}"
)


def ensure_chat_template(tokenizer, template: Optional[str] = None) -> None:
    if tokenizer.chat_template is not None and template is None:
        return
    tokenizer.chat_template = template or DEFAULT_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[data] chat_template set (pad_token={repr(tokenizer.pad_token)})")


def load_sft_dataset(path: str):
    ds = load_dataset("json", data_files=path, split="train")
    if "messages" not in ds.column_names:
        if "instruction" in ds.column_names:
            ds = ds.map(_alpaca_to_messages, remove_columns=ds.column_names)
        else:
            raise ValueError(f"Dataset {path} must have 'messages' or 'instruction' column")
    return ds


def _alpaca_to_messages(example):
    user_content = example.get("instruction", "")
    if example.get("input"):
        user_content += "\n" + example["input"]
    messages = [{"role": "user", "content": user_content}]
    if example.get("output"):
        messages.append({"role": "assistant", "content": example["output"]})
    return {"messages": messages}


def sft_to_bif_pool(train_path: str, output_path: str, source: str = "sft") -> str:
    """Convert SFT JSONL (messages format) to BIF pool_jsonl format."""
    with open(train_path, encoding="utf-8") as f:
        lines = f.readlines()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, line in enumerate(lines):
            record = json.loads(line)
            record_id = record.get("id", str(i))
            messages = record.get("messages", [])

            text_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    text_parts.append(f"### User:\n{content}\n")
                elif role == "assistant":
                    text_parts.append(f"### Assistant:\n{content}\n")

            text = "\n".join(text_parts)
            answer_start = -1
            for msg in messages:
                if msg["role"] == "assistant":
                    answer_start = text.find(msg["content"])
                    break

            bif_record = {
                "id": record_id,
                "source": record.get("source", source),
                "text": text,
            }
            if answer_start >= 0:
                bif_record["answer_start_char"] = answer_start

            f.write(json.dumps(bif_record, ensure_ascii=False) + "\n")

    print(f"[data] BIF pool: {len(lines)} samples -> {output_path}")
    return output_path


def sft_to_bif_query(eval_path: str, output_path: str, source: str = "query") -> str:
    """Convert SFT eval JSONL to BIF query_jsonl format (answer only, with full text and answer_start_char)."""
    with open(eval_path, encoding="utf-8") as f:
        lines = f.readlines()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, line in enumerate(lines):
            record = json.loads(line)
            messages = record.get("messages", [])

            text_parts = []
            answer = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    text_parts.append(f"### User:\n{content}\n")
                elif role == "assistant":
                    answer = content
                    text_parts.append(f"### Assistant:\n{content}\n")

            text = "\n".join(text_parts)
            answer_start = text.find(answer) if answer else -1

            bif_record = {
                "id": record.get("id", f"query_{i}"),
                "source": record.get("source", source),
                "text": text,
            }
            if answer_start >= 0:
                bif_record["answer_start_char"] = answer_start

            f.write(json.dumps(bif_record, ensure_ascii=False) + "\n")

    print(f"[data] BIF query: {len(lines)} samples -> {output_path}")
    return output_path


def split_eval_half(eval_path: str, query_output: str, val_output: str, seed: int = 42) -> dict:
    """Split eval JSONL in half: one for BIF query (question only), one for validation."""
    with open(eval_path, encoding="utf-8") as f:
        lines = f.readlines()

    rng = random.Random(seed)
    indices = list(range(len(lines)))
    rng.shuffle(indices)
    mid = len(indices) // 2
    query_indices = set(indices[:mid])
    val_indices = set(indices[mid:])

    query_lines = [lines[i] for i in sorted(query_indices)]
    val_lines = [lines[i] for i in sorted(val_indices)]

    os.makedirs(os.path.dirname(val_output) or ".", exist_ok=True)
    with open(val_output, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    query_tmp = query_output + ".tmp"
    with open(query_tmp, "w", encoding="utf-8") as f:
        f.writelines(query_lines)
    sft_to_bif_query(query_tmp, query_output, source="query")
    os.remove(query_tmp)

    stats = {"total": len(lines), "query": len(query_lines), "val": len(val_lines)}
    print(f"[data] eval split: {stats['total']} total -> {stats['query']} query, {stats['val']} val")
    return stats


def read_bif_bottom_ids(bif_analysis_dir: str, checkpoint_name: str = "final_model",
                        bottom_k: int = 0) -> list[str]:
    """Read bottom sample IDs from BIF analysis output."""
    scores_csv = os.path.join(bif_analysis_dir, checkpoint_name, "pool_scores.csv")
    bottom_csv = os.path.join(bif_analysis_dir, checkpoint_name, f"bottom_{bottom_k}.csv")

    if bottom_k > 0 and os.path.exists(bottom_csv):
        ids = []
        with open(bottom_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.append(row["sample_id"])
        print(f"[data] BIF bottom_ids from bottom CSV: {len(ids)} ids")
        return ids

    if os.path.exists(scores_csv):
        with open(scores_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        rows.sort(key=lambda r: float(r.get("cross_corr_mean_over_queries", r.get("bif_mean", "0"))))
        k = bottom_k if bottom_k > 0 else len(rows) // 5
        ids = [row["sample_id"] for row in rows[:k]]
        print(f"[data] BIF bottom_ids from pool_scores: {len(ids)} ids (k={k})")
        return ids

    raise FileNotFoundError(f"No BIF analysis files found in {bif_analysis_dir}/{checkpoint_name}/")


def filter_bif_bottom(train_path: str, bottom_ids: list[str], output_path: str) -> dict:
    """Remove bottom samples from training data."""
    bottom_set = set(bottom_ids)

    with open(train_path, encoding="utf-8") as f:
        lines = f.readlines()

    kept, removed = [], 0
    for i, line in enumerate(lines):
        record = json.loads(line)
        record_id = record.get("id", str(i))
        if record_id in bottom_set:
            removed += 1
        else:
            kept.append(line)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(kept)

    stats = {"total": len(lines), "removed": removed, "kept": len(kept)}
    print(f"[filter] removed {removed}, kept {len(kept)} (from {len(lines)} total)")
    return stats


def calc_aligned_epochs(original_epochs: int, original_count: int, filtered_count: int) -> float:
    """Calculate new epoch count to keep total training tokens the same after filtering.

    total_tokens = epochs * samples * avg_tokens_per_sample
    To keep total_tokens constant: new_epochs = original_epochs * original_count / filtered_count
    """
    if filtered_count == 0:
        return original_epochs
    aligned = original_epochs * original_count / filtered_count
    print(f"[align] epochs: {original_epochs} -> {aligned:.2f} (samples: {original_count} -> {filtered_count})")
    return aligned
