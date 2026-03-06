#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

from executors.py_executor import PyExecutor


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_order_map(dataset_rows: List[dict]) -> Dict[str, int]:
    order: Dict[str, int] = {}
    for idx, row in enumerate(dataset_rows):
        for key in ("task_id", "question_id", "entry_point", "prompt"):
            if key in row and row[key] is not None:
                order[str(row[key])] = idx
                break
    return order


def record_rank(rec: dict, order_map: Dict[str, int], fallback: int) -> int:
    for key in ("task_id", "question_id", "entry_point", "prompt"):
        if key in rec and rec[key] is not None:
            return order_map.get(str(rec[key]), fallback)
    return fallback


def recompute_solved(rows: List[dict], timeout: int, include_prompt_context: bool) -> int:
    exe = PyExecutor()
    solved = 0
    for rec in rows:
        entry_point = rec.get("entry_point")
        solution = rec.get("solution")
        test_code = rec.get("test")
        if not (entry_point and isinstance(solution, str) and isinstance(test_code, str)):
            rec["is_solved_reval"] = False
            continue
        candidate = solution
        if include_prompt_context and isinstance(rec.get("prompt"), str):
            candidate = rec["prompt"] + "\n" + solution
        ok = exe.evaluate(entry_point, candidate, test_code, timeout=timeout)
        rec["is_solved_reval"] = ok
        solved += int(ok)
    return solved


def find_worker_log(worker_dir: str) -> str:
    if not os.path.isdir(worker_dir):
        raise FileNotFoundError(f"Missing worker dir: {worker_dir}")

    cands: List[str] = []
    for name in os.listdir(worker_dir):
        if not name.endswith(".jsonl"):
            continue
        p = os.path.join(worker_dir, name)
        cands.append(p)

    if not cands:
        raise FileNotFoundError(f"No jsonl logs found under {worker_dir}")
    if len(cands) == 1:
        return cands[0]

    # Prefer the standard simple-strategy file.
    for p in cands:
        if "_simple_" in os.path.basename(p):
            return p
    return sorted(cands)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--dataset_path", default="benchmarks/humaneval_full.jsonl")
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--eval_timeout", type=int, default=6)
    parser.add_argument(
        "--no_include_prompt_context",
        action="store_false",
        dest="include_prompt_context",
        help="Disable prompt+solution evaluation context (default: enabled).",
    )
    parser.set_defaults(include_prompt_context=True)
    args = parser.parse_args()

    workers_root = os.path.join(args.run_root, "workers")
    merged_root = os.path.join(args.run_root, "merged_baseline")
    os.makedirs(merged_root, exist_ok=True)

    dataset_rows = read_jsonl(args.dataset_path)
    order_map = build_order_map(dataset_rows)

    merged_rows: List[dict] = []
    worker_logs: List[str] = []
    for w in range(args.num_workers):
        worker_name = f"worker_{w:02d}"
        worker_dir = os.path.join(workers_root, worker_name)
        log_path = find_worker_log(worker_dir)
        worker_logs.append(log_path)
        merged_rows.extend(read_jsonl(log_path))

    ranked: List[Tuple[int, dict]] = []
    for i, rec in enumerate(merged_rows):
        ranked.append((record_rank(rec, order_map, len(order_map) + i), rec))
    ranked.sort(key=lambda x: x[0])
    merged_sorted = [r for _, r in ranked]

    # Deduplicate by task_id if present.
    uniq: List[dict] = []
    seen = set()
    for rec in merged_sorted:
        tid = rec.get("task_id")
        if tid is None:
            uniq.append(rec)
            continue
        if tid in seen:
            continue
        seen.add(tid)
        uniq.append(rec)

    out_jsonl = os.path.join(merged_root, "baseline_merged_log.jsonl")
    write_jsonl(out_jsonl, uniq)

    solved = recompute_solved(
        uniq,
        timeout=args.eval_timeout,
        include_prompt_context=args.include_prompt_context,
    )
    total = len(dataset_rows)
    acc = round((solved / total * 100.0), 2) if total else 0.0

    summary = {
        "run_root": args.run_root,
        "dataset_count": total,
        "merged_count": len(uniq),
        "solved": solved,
        "pass_at_1": acc,
        "output_jsonl": out_jsonl,
        "worker_logs": worker_logs,
    }
    summary_path = os.path.join(merged_root, "merge_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    results_txt_path = os.path.join(args.run_root, "results.txt")
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(f"run_root: {args.run_root}\n")
        f.write(f"updated_at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"baseline_solved: {solved}/{total}\n")
        f.write(f"baseline_pass@1: {acc}%\n")
        f.write(f"merged_baseline_dir: {merged_root}\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
