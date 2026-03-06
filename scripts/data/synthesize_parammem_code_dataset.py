#!/usr/bin/env python3
"""Build ParamMem programming supervision data with GPT-4o-mini.

Targets paper-style scale:
- APPS introductory sample: 4000
- Synthetic tasks: 4200

Outputs one JSONL where each line includes:
- source: apps|synth
- sample_id
- func_sign
- docstring
- pitfalls (list[str])
- buggy_implementations (list[str])
- category (for synth)
- meta
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_keys() -> List[str]:
    keys: List[str] = []
    for i in range(2, 26):
        k = os.getenv(f"ZZZ_API_KEY_{i}")
        if k:
            keys.append(k)
    if not keys and os.getenv("OPENAI_API_KEY"):
        keys.append(os.getenv("OPENAI_API_KEY", ""))
    uniq: List[str] = []
    seen = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


def load_apps_from_jsonl(path: str) -> List[dict]:
    rows = read_jsonl(path)
    return rows


def load_apps_from_hf(split: str) -> List[dict]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "datasets is required for HF loading. Install with: pip install datasets"
        ) from e
    rows: List[dict] = []
    splits = [s.strip() for s in split.split("+") if s.strip()]
    if not splits:
        splits = ["train"]
    for sp in splits:
        ds = load_dataset("codeparrot/apps", split=sp)
        for x in ds:
            rec = dict(x)
            rec["__split"] = sp
            rows.append(rec)
    return rows


def extract_apps_prompt(item: dict) -> str:
    candidates = [
        "question",
        "problem_statement",
        "prompt",
        "description",
    ]
    for k in candidates:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    text = json.dumps(item, ensure_ascii=False)
    return text[:8000]


def filter_intro_apps(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    for r in rows:
        diff = str(r.get("difficulty", "")).lower()
        if "intro" in diff:
            out.append(r)
    if out:
        return out
    return rows


def stable_id(prefix: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


SYNTH_CATEGORIES = [
    "String Manipulation",
    "Array / Two Pointers",
    "Hashing / Counting",
    "Sorting / Greedy",
    "Recursion / Backtracking",
    "Dynamic Programming",
    "Graph / BFS / DFS",
    "Tree / Binary Tree",
    "Math / Number Theory",
    "Interval / Sweep Line",
    "Simulation / Game Logic",
    "Misc Small-Scale Algorithms",
]


@dataclass
class Task:
    source: str
    sample_id: str
    category: str
    prompt_text: str


def build_tasks(
    apps_rows: List[dict],
    apps_samples: int,
    synth_samples: int,
    seed: int,
) -> List[Task]:
    rng = random.Random(seed)
    intro_rows = filter_intro_apps(apps_rows)
    rng.shuffle(intro_rows)
    selected_apps = intro_rows[:apps_samples]
    if len(selected_apps) < apps_samples:
        print(
            f"[warn] requested apps_samples={apps_samples}, but only {len(selected_apps)} intro samples are available."
        )

    tasks: List[Task] = []
    for row in selected_apps:
        p = extract_apps_prompt(row)
        split_name = str(row.get("__split", "unknown"))
        pid = row.get("problem_id") or row.get("task_id")
        if pid is not None:
            sid = f"{split_name}_{pid}"
        else:
            sid = stable_id(f"apps_{split_name}", p)
        tasks.append(Task(source="apps", sample_id=str(sid), category="", prompt_text=p))

    for i in range(synth_samples):
        cat = SYNTH_CATEGORIES[i % len(SYNTH_CATEGORIES)]
        sid = f"synth_{i:05d}"
        tasks.append(Task(source="synth", sample_id=sid, category=cat, prompt_text=""))

    return tasks


def extract_json_obj(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    chunk = m.group(0)
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        return None


def validate_record(obj: dict) -> Tuple[bool, str]:
    required = ["func_sign", "docstring", "pitfalls", "buggy_implementations"]
    for k in required:
        if k not in obj:
            return False, f"missing:{k}"
    if not isinstance(obj["func_sign"], str) or not obj["func_sign"].strip().startswith("def "):
        return False, "bad_func_sign"
    if not isinstance(obj["docstring"], str) or len(obj["docstring"].strip()) < 20:
        return False, "bad_docstring"
    if not isinstance(obj["pitfalls"], list) or len(obj["pitfalls"]) < 4:
        return False, "bad_pitfalls"
    if not isinstance(obj["buggy_implementations"], list) or len(obj["buggy_implementations"]) < 4:
        return False, "bad_buggy_impls"
    return True, "ok"


def build_messages(task: Task) -> List[dict]:
    system = (
        "You are an expert Python educator creating high-quality reflective supervision data for code agents. "
        "Return strict JSON only."
    )
    if task.source == "apps":
        user = (
            "Given the coding problem below, produce JSON with keys: "
            "func_sign, docstring, pitfalls, buggy_implementations.\n"
            "Requirements:\n"
            "1) func_sign must be one python function signature ending with colon.\n"
            "2) docstring must specify behavior, edge cases, and constraints.\n"
            "3) pitfalls must contain 4-8 concrete implementation mistakes.\n"
            "4) buggy_implementations must contain 5 complete python function implementations, each with a different realistic bug.\n"
            "5) Keep all buggy implementations aligned to func_sign and docstring.\n"
            "6) Output valid JSON only, no markdown.\n\n"
            f"[Coding Problem]\n{task.prompt_text[:12000]}"
        )
    else:
        user = (
            "Create a new Python coding task in category: "
            f"{task.category}.\n"
            "Output JSON with keys func_sign, docstring, pitfalls, buggy_implementations.\n"
            "Requirements:\n"
            "1) The task should be non-trivial but solvable (LeetCode easy/medium style).\n"
            "2) func_sign is one signature ending with colon.\n"
            "3) docstring includes edge cases and expected complexity.\n"
            "4) pitfalls has 4-8 concrete mistakes.\n"
            "5) buggy_implementations has 5 complete wrong implementations, each illustrating different pitfalls.\n"
            "6) Output valid JSON only, no markdown."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_model(
    client: OpenAI,
    model: str,
    messages: List[dict],
    temperature: float,
    max_tokens: int,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def worker_run(
    worker_id: int,
    key: str,
    base_url: str,
    model: str,
    tasks: List[Task],
    out_dir: str,
    max_retries: int,
    temperature: float,
    max_tokens: int,
    error_raw_max_chars: int,
) -> dict:
    client = OpenAI(api_key=key, base_url=base_url)
    shard_path = os.path.join(out_dir, f"worker_{worker_id:02d}.jsonl")
    err_path = os.path.join(out_dir, f"worker_{worker_id:02d}_errors.jsonl")

    ok = 0
    fail = 0
    for t in tasks:
        result = None
        last_err = ""
        last_raw = ""
        for attempt in range(max_retries):
            try:
                msgs = build_messages(t)
                raw = call_model(
                    client=client,
                    model=model,
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                last_raw = raw
                obj = extract_json_obj(raw)
                if obj is None:
                    last_err = "json_parse_failed"
                    continue
                valid, reason = validate_record(obj)
                if not valid:
                    last_err = reason
                    continue

                result = {
                    "source": t.source,
                    "sample_id": t.sample_id,
                    "category": t.category,
                    "func_sign": obj["func_sign"].strip(),
                    "docstring": obj["docstring"].strip(),
                    "pitfalls": obj["pitfalls"],
                    "buggy_implementations": obj["buggy_implementations"],
                    "meta": {
                        "worker": worker_id,
                        "attempt": attempt + 1,
                        "model": model,
                        "temperature": temperature,
                    },
                }
                break
            except Exception as e:  # noqa: BLE001
                last_err = f"exception:{type(e).__name__}:{e}"
                time.sleep(0.6 * (attempt + 1))

        if result is not None:
            append_jsonl(shard_path, result)
            ok += 1
        else:
            append_jsonl(
                err_path,
                {
                    "source": t.source,
                    "sample_id": t.sample_id,
                    "category": t.category,
                    "error": last_err,
                    "raw_response": (last_raw[:error_raw_max_chars] if error_raw_max_chars > 0 else ""),
                },
            )
            fail += 1

    return {"worker": worker_id, "ok": ok, "fail": fail, "shard": shard_path, "err": err_path}


def merge_outputs(shard_paths: List[str], output_jsonl: str) -> int:
    rows: Dict[str, dict] = {}
    for p in shard_paths:
        if not os.path.exists(p):
            continue
        for r in read_jsonl(p):
            key = str(r.get("sample_id"))
            rows[key] = r
    merged = list(rows.values())
    merged.sort(key=lambda x: x.get("sample_id", ""))
    write_jsonl(output_jsonl, merged)
    return len(merged)


def build_quality_report(path: str) -> dict:
    rows = read_jsonl(path)
    total = len(rows)
    src_counts: Dict[str, int] = {}
    pit_empty = 0
    buggy_empty = 0
    for r in rows:
        src = str(r.get("source", "unknown"))
        src_counts[src] = src_counts.get(src, 0) + 1
        if not r.get("pitfalls"):
            pit_empty += 1
        if not r.get("buggy_implementations"):
            buggy_empty += 1
    return {
        "total": total,
        "source_counts": src_counts,
        "pitfalls_empty": pit_empty,
        "buggy_empty": buggy_empty,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apps_source_jsonl", default="", help="Optional local APPS JSONL path")
    parser.add_argument("--apps_hf_split", default="train", help="HF split for codeparrot/apps")
    parser.add_argument("--apps_samples", type=int, default=4000)
    parser.add_argument("--synth_samples", type=int, default=4200)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--error_raw_max_chars", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--work_dir", default="data/train/parammem_build")
    args = parser.parse_args()

    keys = collect_keys()
    if not keys:
        raise RuntimeError("No API keys found. Set ZZZ_API_KEY_2..25 or OPENAI_API_KEY.")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.zhizengzeng.com/v1")

    if args.apps_source_jsonl:
        apps_rows = load_apps_from_jsonl(args.apps_source_jsonl)
    else:
        apps_rows = load_apps_from_hf(args.apps_hf_split)

    tasks = build_tasks(
        apps_rows=apps_rows,
        apps_samples=args.apps_samples,
        synth_samples=args.synth_samples,
        seed=args.seed,
    )

    workers = min(args.workers, len(keys))
    key_pool = keys[:workers]

    shard_buckets: List[List[Task]] = [[] for _ in range(workers)]
    for idx, t in enumerate(tasks):
        shard_buckets[idx % workers].append(t)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.work_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    futures = []
    stats: List[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i in range(workers):
            futures.append(
                ex.submit(
                    worker_run,
                    i,
                    key_pool[i],
                    base_url,
                    args.model,
                    shard_buckets[i],
                    run_dir,
                    args.max_retries,
                    args.temperature,
                    args.max_tokens,
                    args.error_raw_max_chars,
                )
            )
        for fut in as_completed(futures):
            stats.append(fut.result())

    shard_paths = [s["shard"] for s in stats]
    merged_count = merge_outputs(shard_paths, args.output_jsonl)
    report = build_quality_report(args.output_jsonl)

    meta = {
        "timestamp": ts,
        "output_jsonl": args.output_jsonl,
        "run_dir": run_dir,
        "apps_samples": args.apps_samples,
        "synth_samples": args.synth_samples,
        "target_total": args.apps_samples + args.synth_samples,
        "merged_total": merged_count,
        "workers": workers,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": 1.0,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "max_retries": args.max_retries,
        "error_raw_max_chars": args.error_raw_max_chars,
        "stats": sorted(stats, key=lambda x: x["worker"]),
        "quality": report,
    }

    meta_path = os.path.join(os.path.dirname(args.output_jsonl) or ".", "synthesis_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
