import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import read_jsonl


def summarize(run_dir: str) -> None:
    first_path = os.path.join(run_dir, "first_stage_log.jsonl")
    second_path = os.path.join(run_dir, "second_stage_log.jsonl")

    if not os.path.exists(first_path):
        raise FileNotFoundError(f"Missing: {first_path}")

    first = read_jsonl(first_path)
    solved1 = sum(1 for x in first if x.get("is_solved", False))
    print(f"Phase 1: {solved1}/{len(first)} = {solved1/len(first)*100:.2f}%")

    rows_for_cost = first
    if os.path.exists(second_path):
        second = read_jsonl(second_path)
        solved2 = sum(1 for x in second if x.get("is_solved", False))
        print(f"Phase 2: {solved2}/{len(second)} = {solved2/len(second)*100:.2f}%")
        rows_for_cost = second
    else:
        print("Phase 2: second_stage_log.jsonl not found yet")

    total_cost = sum(x.get("cost", 0.0) for x in rows_for_cost)
    pt = sum(x.get("prompt_tokens", 0) for x in rows_for_cost)
    ct = sum(x.get("completion_tokens", 0) for x in rows_for_cost)
    print(f"Prompt tokens: {pt:,}")
    print(f"Completion tokens: {ct:,}")
    print(f"Total cost: ${total_cost:.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/hpc4/check_pathA_results.py <run_dir>")
        sys.exit(1)
    summarize(sys.argv[1])
