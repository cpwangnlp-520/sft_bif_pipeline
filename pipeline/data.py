from __future__ import annotations

import csv
import json
import os
import random
from typing import Optional

from datasets import load_dataset

from .config import DropConfig


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


def prepare_bif_query(eval_files: dict[str, str], exclude: list[str], output_path: str) -> str:
    query_records = []
    for name, path in eval_files.items():
        if name in exclude:
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                query_records.append(line)

    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(query_records)

    sft_to_bif_query(tmp_path, output_path, source="eval")
    os.remove(tmp_path)
    return output_path


def read_bif_ids(csv_dir: str, csv_name: str, which: str, k: int) -> list[str]:
    csv_path = os.path.join(csv_dir, csv_name, f"{which}_{k}.csv")
    if os.path.exists(csv_path):
        ids = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "sample_id" in row:
                    ids.append(row["sample_id"])
        print(f"[data] Read {len(ids)} {which} IDs from {csv_path}")
        return ids

    pool_path = os.path.join(csv_dir, csv_name, "pool_scores.csv")
    if os.path.exists(pool_path):
        with open(pool_path, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        score_col = "cross_corr_mean_over_queries"
        reader.sort(key=lambda r: float(r.get(score_col, r.get("bif_mean", "0"))),
                    reverse=(which == "top"))
        ids = [row["sample_id"] for row in reader[:k]]
        print(f"[data] Read {len(ids)} {which} IDs from pool_scores (k={k})")
        return ids

    raise FileNotFoundError(f"No {which}_{k}.csv or pool_scores.csv in {csv_dir}/{csv_name}/")


def _drop_ids_from_jsonl(train_path: str, drop_ids: set[str], output_path: str) -> dict:
    with open(train_path, encoding="utf-8") as f:
        lines = f.readlines()

    kept, removed = [], 0
    for i, line in enumerate(lines):
        record = json.loads(line)
        record_id = str(record.get("id", str(i)))
        if record_id in drop_ids:
            removed += 1
        else:
            kept.append(line)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(kept)

    stats = {"total": len(lines), "removed": removed, "kept": len(kept)}
    print(f"[drop] {len(lines)} -> removed {removed}, kept {len(kept)} -> {output_path}")
    return stats


def prepare_drop_data(
    train_path: str,
    drop_cfg: DropConfig,
    output_dir: str,
) -> dict[str, str]:
    with open(train_path, encoding="utf-8") as f:
        train_records = [json.loads(line) for line in f]

    pool_ids = None
    if drop_cfg.pool_only and drop_cfg.pool_size > 0:
        pool_ids = set()
        for rec in train_records[:drop_cfg.pool_size]:
            pool_ids.add(str(rec.get("id", "")))

    drop_id_sets = {}
    if "bottom" in drop_cfg.strategies:
        drop_id_sets["bottom"] = set(read_bif_ids(drop_cfg.csv_dir, drop_cfg.csv_name, "bottom", drop_cfg.k))
    if "top" in drop_cfg.strategies:
        drop_id_sets["top"] = set(read_bif_ids(drop_cfg.csv_dir, drop_cfg.csv_name, "top", drop_cfg.k))
    if "random" in drop_cfg.strategies:
        if pool_ids is not None:
            candidates = list(pool_ids)
        else:
            candidates = [str(rec.get("id", str(i))) for i, rec in enumerate(train_records)]
        rng = random.Random(drop_cfg.random_seed)
        drop_id_sets["random"] = set(rng.sample(candidates, min(drop_cfg.k, len(candidates))))
        print(f"[data] Random drop: {len(drop_id_sets['random'])} IDs (pool_only={drop_cfg.pool_only})")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(train_path))[0]
    results = {}

    for name, ids in drop_id_sets.items():
        out_path = os.path.join(output_dir, f"{base_name}_drop_{name}_{drop_cfg.k}.jsonl")
        _drop_ids_from_jsonl(train_path, ids, out_path)
        results[name] = out_path

    return results
