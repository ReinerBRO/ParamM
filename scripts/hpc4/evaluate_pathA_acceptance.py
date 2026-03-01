#!/usr/bin/env python3
import json
import os
import pickle
import sys
from typing import Dict, List


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def pass_rate(rows: List[dict]) -> float:
    if not rows:
        return 0.0
    solved = sum(1 for x in rows if x.get("is_solved", False))
    return solved / len(rows) * 100.0


def all_fields_present(rows: List[dict], fields: List[str]) -> bool:
    for item in rows:
        for f in fields:
            if f not in item:
                return False
    return True


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: evaluate_pathA_acceptance.py <run_dir> <nohup_log>")
        return 2

    run_dir = sys.argv[1]
    nohup_log = sys.argv[2]
    first = os.path.join(run_dir, "first_stage_log.jsonl")
    second = os.path.join(run_dir, "second_stage_log.jsonl")
    mem = os.path.join(run_dir, "mem_bank.pkl")

    result: Dict[str, object] = {
        "run_dir": run_dir,
        "nohup_log": nohup_log,
        "hard_pass": {},
        "soft_pass": {},
        "metrics": {},
    }

    if not os.path.exists(first):
        result["error"] = "first_stage_log.jsonl missing"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    first_rows = read_jsonl(first)
    second_rows = read_jsonl(second) if os.path.exists(second) else []
    final_rows = second_rows if second_rows else first_rows

    p1 = pass_rate(first_rows)
    p2 = pass_rate(second_rows) if second_rows else p1
    prompt_tokens = sum(int(x.get("prompt_tokens", 0)) for x in final_rows)
    completion_tokens = sum(int(x.get("completion_tokens", 0)) for x in final_rows)
    cost = sum(float(x.get("cost", 0.0)) for x in final_rows)

    result["metrics"] = {
        "phase1_count": len(first_rows),
        "phase2_count": len(second_rows),
        "phase1_pass1": p1,
        "phase2_pass1": p2,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": cost,
    }

    required_fields = [
        "is_solved",
        "solution",
        "cost",
        "prompt_tokens",
        "completion_tokens",
        "diverse_reflections",
        "implementations",
    ]

    hard = {}
    hard["H1_phase1_164"] = len(first_rows) == 164
    hard["H2_phase2_164"] = len(second_rows) == 164
    hard["H3_fields_complete"] = all_fields_present(final_rows, required_fields)
    mem_ok = False
    if os.path.exists(mem):
        try:
            with open(mem, "rb") as f:
                mb = pickle.load(f)
            mem_ok = (
                isinstance(mb, dict)
                and "positive_trajectories" in mb
                and "negative_trajectories" in mb
            )
        except Exception:
            mem_ok = False
    hard["H4_memory_bank_ok"] = mem_ok
    log_ok = True
    if os.path.exists(nohup_log):
        try:
            txt = open(nohup_log, "r", encoding="utf-8", errors="ignore").read()
            # best-effort: any traceback or uncaught exception indicates issue
            if "Traceback (most recent call last)" in txt:
                log_ok = False
        except Exception:
            pass
    hard["H5_no_traceback"] = log_ok
    result["hard_pass"] = hard

    soft = {}
    soft["S1_phase1_75_90"] = 75.0 <= p1 <= 90.0
    soft["S2_phase2_80_95"] = 80.0 <= p2 <= 95.0
    soft["S3_phase2_ge_phase1"] = p2 >= p1
    soft["S4_phase1_gt_60"] = p1 > 60.0
    soft["S5_prompt_tokens_400k_2m"] = 400_000 <= prompt_tokens <= 2_000_000
    result["soft_pass"] = soft

    result["hard_pass_all"] = all(hard.values())
    result["soft_pass_all"] = all(soft.values())
    result["acceptance_reached"] = bool(result["hard_pass_all"] and result["soft_pass_all"])

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
