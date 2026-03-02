#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple


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


def find_worker_log(worker_dir: str) -> str:
    for name in os.listdir(worker_dir):
        if not name.endswith(".jsonl"):
            continue
        if name in {"first_stage_log.jsonl", "second_stage_log.jsonl"}:
            continue
        return os.path.join(worker_dir, name)
    raise FileNotFoundError(f"No main jsonl log found in {worker_dir}")


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


def merge_failed(paths: List[str]) -> List[dict]:
    out: List[dict] = []
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            out.extend(pickle.load(f))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, help="Run root created by run_phase1_sharded_miku.sh")
    parser.add_argument("--dataset_path", default="benchmarks/humaneval_full.jsonl")
    parser.add_argument("--num_workers", type=int, default=24)
    args = parser.parse_args()

    workers_root = os.path.join(args.run_root, "workers")
    merged_root = os.path.join(args.run_root, "merged_phase1")
    os.makedirs(merged_root, exist_ok=True)

    dataset_rows = read_jsonl(args.dataset_path)
    order_map = build_order_map(dataset_rows)

    merged_first_stage: List[dict] = []
    merged_main_log: List[dict] = []
    mem_paths: List[str] = []
    failed_paths: List[str] = []

    for w in range(args.num_workers):
        worker_name = f"worker_{w:02d}"
        worker_dir = os.path.join(workers_root, worker_name)
        if not os.path.isdir(worker_dir):
            raise FileNotFoundError(f"Missing worker dir: {worker_dir}")

        first_stage_path = os.path.join(worker_dir, "first_stage_log.jsonl")
        if not os.path.exists(first_stage_path):
            raise FileNotFoundError(f"Missing first_stage_log.jsonl: {first_stage_path}")

        merged_first_stage.extend(read_jsonl(first_stage_path))

        main_log_path = find_worker_log(worker_dir)
        merged_main_log.extend(read_jsonl(main_log_path))

        mem_paths.append(os.path.join(worker_dir, "mem_bank.pkl"))
        failed_paths.append(os.path.join(worker_dir, "failed_probs.pkl"))

    ranked_first: List[Tuple[int, dict]] = []
    for i, rec in enumerate(merged_first_stage):
        ranked_first.append((record_rank(rec, order_map, len(order_map) + i), rec))
    ranked_first.sort(key=lambda x: x[0])
    merged_first_stage_sorted = [rec for _, rec in ranked_first]

    ranked_main: List[Tuple[int, dict]] = []
    for i, rec in enumerate(merged_main_log):
        ranked_main.append((record_rank(rec, order_map, len(order_map) + i), rec))
    ranked_main.sort(key=lambda x: x[0])
    merged_main_sorted = [rec for _, rec in ranked_main]

    merged_mem = merge_mem_banks(mem_paths)
    merged_failed = merge_failed(failed_paths)

    first_out = os.path.join(merged_root, "first_stage_log.jsonl")
    main_out = os.path.join(merged_root, "phase1_merged_log.jsonl")
    mem_out = os.path.join(merged_root, "mem_bank.pkl")
    failed_out = os.path.join(merged_root, "failed_probs.pkl")
    summary_out = os.path.join(merged_root, "merge_summary.json")

    write_jsonl(first_out, merged_first_stage_sorted)
    write_jsonl(main_out, merged_main_sorted)
    with open(mem_out, "wb") as f:
        pickle.dump(merged_mem, f)
    with open(failed_out, "wb") as f:
        pickle.dump(merged_failed, f)

    summary = {
        "num_workers": args.num_workers,
        "dataset_count": len(dataset_rows),
        "merged_first_stage_count": len(merged_first_stage_sorted),
        "merged_main_log_count": len(merged_main_sorted),
        "merged_positive_trajectories": len(merged_mem.get("positive_trajectories", [])),
        "merged_negative_trajectories": len(merged_mem.get("negative_trajectories", [])),
        "merged_failed_count": len(merged_failed),
        "output_dir": merged_root,
        "first_stage_log": first_out,
        "phase1_merged_log": main_out,
        "mem_bank": mem_out,
        "failed_probs": failed_out,
    }
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
