#!/usr/bin/env python3
import argparse
import json
import os
import pickle
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


def merge_mem_banks(paths: List[str]) -> dict:
    merged = {"positive_trajectories": [], "negative_trajectories": []}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            mb = pickle.load(f)
        merged["positive_trajectories"].extend(mb.get("positive_trajectories", []))
        merged["negative_trajectories"].extend(mb.get("negative_trajectories", []))
    return merged


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--dataset_path", default="benchmarks/humaneval_full.jsonl")
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--eval_timeout", type=int, default=10)
    parser.add_argument(
        "--no_include_prompt_context",
        action="store_false",
        dest="include_prompt_context",
        help="Disable prompt+solution evaluation context (default: enabled).",
    )
    parser.set_defaults(include_prompt_context=True)
    args = parser.parse_args()

    workers_root = os.path.join(args.run_root, "workers")
    merged_root = os.path.join(args.run_root, "merged_phase2")
    os.makedirs(merged_root, exist_ok=True)

    dataset_rows = read_jsonl(args.dataset_path)
    order_map = build_order_map(dataset_rows)

    merged_first: List[dict] = []
    merged_second: List[dict] = []
    mem_paths: List[str] = []

    for w in range(args.num_workers):
        worker_name = f"worker_{w:02d}"
        worker_dir = os.path.join(workers_root, worker_name)
        if not os.path.isdir(worker_dir):
            raise FileNotFoundError(f"Missing worker dir: {worker_dir}")

        first_stage_path = os.path.join(worker_dir, "first_stage_log.jsonl")
        second_stage_path = os.path.join(worker_dir, "second_stage_log.jsonl")
        if not os.path.exists(first_stage_path):
            raise FileNotFoundError(f"Missing first_stage_log: {first_stage_path}")

        first_rows = read_jsonl(first_stage_path)
        merged_first.extend(first_rows)
        if os.path.exists(second_stage_path):
            merged_second.extend(read_jsonl(second_stage_path))
        else:
            # Worker had no phase2 candidates; phase2 output is identical to phase1.
            merged_second.extend(first_rows)
        mem_paths.append(os.path.join(worker_dir, "mem_bank.pkl"))

    ranked_first: List[Tuple[int, dict]] = []
    for i, rec in enumerate(merged_first):
        ranked_first.append((record_rank(rec, order_map, len(order_map) + i), rec))
    ranked_first.sort(key=lambda x: x[0])
    merged_first_sorted = [rec for _, rec in ranked_first]

    ranked_second: List[Tuple[int, dict]] = []
    for i, rec in enumerate(merged_second):
        ranked_second.append((record_rank(rec, order_map, len(order_map) + i), rec))
    ranked_second.sort(key=lambda x: x[0])
    merged_second_sorted = [rec for _, rec in ranked_second]

    merged_mem = merge_mem_banks(mem_paths)

    first_out = os.path.join(merged_root, "first_stage_log.jsonl")
    second_out = os.path.join(merged_root, "second_stage_log.jsonl")
    mem_out = os.path.join(merged_root, "mem_bank.pkl")
    summary_out = os.path.join(merged_root, "merge_summary.json")

    write_jsonl(first_out, merged_first_sorted)
    write_jsonl(second_out, merged_second_sorted)
    with open(mem_out, "wb") as f:
        pickle.dump(merged_mem, f)

    phase1_solved = recompute_solved(
        merged_first_sorted,
        timeout=args.eval_timeout,
        include_prompt_context=args.include_prompt_context,
    )
    phase2_solved = recompute_solved(
        merged_second_sorted,
        timeout=args.eval_timeout,
        include_prompt_context=args.include_prompt_context,
    )

    summary = {
        "num_workers": args.num_workers,
        "dataset_count": len(dataset_rows),
        "phase1_count": len(merged_first_sorted),
        "phase2_count": len(merged_second_sorted),
        "phase1_solved": phase1_solved,
        "phase2_solved": phase2_solved,
        "phase1_acc": round((phase1_solved / len(merged_first_sorted) * 100.0), 2) if merged_first_sorted else 0.0,
        "phase2_acc": round((phase2_solved / len(merged_second_sorted) * 100.0), 2) if merged_second_sorted else 0.0,
        "merged_positive_trajectories": len(merged_mem.get("positive_trajectories", [])),
        "merged_negative_trajectories": len(merged_mem.get("negative_trajectories", [])),
        "output_dir": merged_root,
        "first_stage_log": first_out,
        "second_stage_log": second_out,
        "mem_bank": mem_out,
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    results_txt_path = os.path.join(args.run_root, "results.txt")
    with open(results_txt_path, "w", encoding="utf-8") as f:
        f.write(f"run_root: {args.run_root}\n")
        f.write(f"updated_at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"phase1_solved: {phase1_solved}/{len(dataset_rows)}\n")
        f.write(f"phase1_acc: {summary['phase1_acc']}%\n")
        f.write(f"phase2_solved: {phase2_solved}/{len(dataset_rows)}\n")
        f.write(f"phase2_acc: {summary['phase2_acc']}%\n")
        f.write(f"merged_phase2_dir: {merged_root}\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
