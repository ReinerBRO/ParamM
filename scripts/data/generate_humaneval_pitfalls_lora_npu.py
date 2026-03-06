#!/usr/bin/env python3
"""Generate HumanEval pitfalls using local Llama-3.1-8B + LoRA on one NPU shard."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = (
    "You are an AI assistant in coding. "
    "Given a Python function signature and docstring, list potential pitfalls.\n\n"
    "[Example]\n"
    "def longest_subarray_with_sum_limit(nums: List[int], target: int) -> List[int]:\n"
    "    \"\"\"\n"
    "    Return the longest **contiguous** subarray of `nums` whose elements sum to at most `target`.\n"
    "\n"
    "    • If several subarrays tie for maximum length, return the **left‑most**.\n"
    "    • If no valid subarray exists, return the empty list `[]`.\n"
    "    • The input may contain negative as well as positive integers.\n"
    "\n"
    "    Complexity requirements: time O(n), auxiliary space O(1).\n"
    "    \"\"\"\n"
    "\n"
    "[Pitfalls]:\n"
    "1. **No‑solution case** — must return `[]`, not `[x]` or `None`.\n"
    "2. **Length update rule** — use strictly greater (`>`); otherwise, a later equal‑length window overwrites the earlier left‑most one.\n"
    "3. **Negatives in the window** — shrinking only while `current_sum > target` can leave an over‑target sum if later negatives cancel it.\n"
    "\n"
    "Now, list potential pitfalls for the following question:"
)


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def row_uid(row: dict) -> str:
    for key in ("task_id", "name", "question_id", "entry_point", "prompt"):
        v = row.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def format_prompt(prompt_text: str) -> str:
    user_block = (
        "[INST] <<SYS>> "
        f"{SYSTEM_PROMPT} <</SYS>>\n\n"
        f"FUNC_SIGNATURE:\n{prompt_text.strip()}\n\n"
        "[/INST]"
    )
    return f"<s>{user_block}"


def clean_reply(text: str) -> str:
    reply = text.split("[/INST]", 1)[-1].strip()
    reply = re.sub(r"</s>|<s>|\[/?INST\]", "", reply).strip()
    fence_pos = reply.rfind("```")
    if fence_pos != -1:
        reply = reply[: fence_pos + 3]
    return reply.strip()


def generate_one(model, tokenizer, device, prompt_text: str, temperature: float, max_new_tokens: int) -> str:
    prompt = format_prompt(prompt_text)
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=1536,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    do_sample = temperature > 0.0
    with torch.inference_mode():
        outs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    decoded = tokenizer.batch_decode(outs, skip_special_tokens=False)[0]
    return clean_reply(decoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--shard_total", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    args = parser.parse_args()

    import torch_npu  # noqa: F401

    device = torch.device("npu:0")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model = PeftModel.from_pretrained(base, args.lora_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    model.config.use_cache = True

    rows = read_jsonl(args.input_jsonl)
    shard_rows = [r for i, r in enumerate(rows) if i % args.shard_total == args.shard_idx]

    done_map = {}
    if os.path.exists(args.output_jsonl):
        for r in read_jsonl(args.output_jsonl):
            uid = row_uid(r)
            if uid:
                done_map[uid] = r

    out = list(done_map.values())
    total = len(shard_rows)
    for idx, item in enumerate(shard_rows, 1):
        uid = row_uid(item)
        if uid and uid in done_map:
            continue
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            continue

        pitfall = generate_one(model, tokenizer, device, prompt, temperature=0.2, max_new_tokens=args.max_new_tokens)
        high_temp = generate_one(model, tokenizer, device, prompt, temperature=1.0, max_new_tokens=args.max_new_tokens)

        rec = dict(item)
        rec["pitfall"] = pitfall
        rec["high_temp_pitfall"] = high_temp
        out.append(rec)

        write_jsonl(args.output_jsonl, sorted(out, key=row_uid))
        print(f"[shard {args.shard_idx}] {idx}/{total} written={len(out)}", flush=True)

    write_jsonl(args.output_jsonl, sorted(out, key=row_uid))
    print(f"[shard {args.shard_idx}] done total={len(out)}", flush=True)


if __name__ == "__main__":
    main()
