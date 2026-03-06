"""Build training dataset for Transformer router from phase1/phase2 logs.

This script converts existing phase logs into the format needed for
training the Transformer router with enhanced features.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping

from memory_router.feature_schema import normalize_mix


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file."""
    rows: List[Dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping):
                rows.append(dict(row))
    return rows


def _write_jsonl(path: str, records: List[Mapping[str, Any]]) -> int:
    """Write JSONL file."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _extract_sims(record: Mapping[str, Any], key_prefix: str) -> List[float]:
    """Extract similarity scores from record."""
    for key in [f"{key_prefix}_sims", f"{key_prefix}_sim", f"{key_prefix}_similarity"]:
        val = record.get(key)
        if isinstance(val, list):
            return [float(x) for x in val if x is not None]
    return []


def _build_training_sample(phase1: Mapping[str, Any], phase2: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a training sample from phase1 and phase2 logs.

    Args:
        phase1: Phase1 log entry
        phase2: Phase2 log entry

    Returns:
        Training sample dictionary
    """
    # Merge state from both phases
    reflections = phase2.get("diverse_reflections") or phase1.get("diverse_reflections", [])
    if not isinstance(reflections, list):
        reflections = []

    test_feedback = phase2.get("test_feedback") or phase1.get("test_feedback", [])
    if not isinstance(test_feedback, list):
        test_feedback = []

    implementations = phase2.get("implementations") or phase1.get("implementations", [])
    if not isinstance(implementations, list):
        implementations = []

    state = {
        "prompt": phase2.get("prompt") or phase1.get("prompt", ""),
        "reflections": reflections,
        "test_feedback": test_feedback,
        "attempt_count": len(implementations),
        "reflection_rounds": len(reflections),
    }

    # Extract similarity scores (may be empty if not logged)
    prompt_sims = _extract_sims(phase2, "prompt") or _extract_sims(phase1, "prompt")
    reflection_sims = _extract_sims(phase2, "reflection") or _extract_sims(phase1, "reflection")
    negative_sims = _extract_sims(phase2, "negative") or _extract_sims(phase1, "negative")

    # If no similarity scores, create mock ones based on attempt count
    if not prompt_sims:
        # Create 3-5 mock candidates with decreasing similarity
        n_candidates = min(5, max(3, len(implementations)))
        prompt_sims = [0.8 - 0.1 * i for i in range(n_candidates)]
        reflection_sims = [0.6 - 0.1 * i for i in range(n_candidates)]
        negative_sims = [0.1 + 0.05 * i for i in range(n_candidates)]

    # Determine outcome
    outcome_solved = 0
    for key in ["solved", "pass", "passed", "is_solved", "success"]:
        val = phase2.get(key) or phase1.get(key)
        if val is not None:
            outcome_solved = int(bool(val))
            break

    # Heuristic target mix (similar to original dataset_builder)
    prompt_signal = max(prompt_sims) if prompt_sims else 0.0
    reflection_signal = max(reflection_sims) if reflection_sims else 0.0
    negative_signal = max(negative_sims) if negative_sims else 0.0

    w_prompt = 0.30 + prompt_signal
    w_reflection = 0.25 + reflection_signal + 0.06 * state["reflection_rounds"]
    w_negative = 0.20 + negative_signal

    if outcome_solved:
        w_prompt += 0.25
        w_reflection += 0.20
    else:
        w_negative += 0.35

    target_mix = normalize_mix([w_prompt, w_reflection, w_negative])

    return {
        "state": state,
        "prompt_sims": prompt_sims,
        "reflection_sims": reflection_sims,
        "negative_sims": negative_sims,
        "outcome_solved": outcome_solved,
        "target_mix": target_mix,
        "task_id": phase2.get("task_id") or phase1.get("task_id", "unknown"),
    }


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    """Build training dataset from phase logs."""
    print("Loading phase1 logs...")
    phase1_records = _read_jsonl(args.phase1_jsonl)
    print(f"Loaded {len(phase1_records)} phase1 records")

    print("Loading phase2 logs...")
    phase2_records = _read_jsonl(args.phase2_jsonl)
    print(f"Loaded {len(phase2_records)} phase2 records")

    # Build task_id -> record maps
    phase1_map = {r.get("task_id", f"idx_{i}"): r for i, r in enumerate(phase1_records)}
    phase2_map = {r.get("task_id", f"idx_{i}"): r for i, r in enumerate(phase2_records)}

    # Merge and build samples
    all_task_ids = sorted(set(phase1_map.keys()) | set(phase2_map.keys()))
    samples: List[Dict[str, Any]] = []

    print("Building training samples...")
    for task_id in all_task_ids:
        phase1 = phase1_map.get(task_id, {})
        phase2 = phase2_map.get(task_id, {})

        if not phase1 and not phase2:
            continue

        try:
            sample = _build_training_sample(phase1, phase2)
            samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed to build sample for {task_id}: {e}")
            continue

    print(f"Built {len(samples)} training samples")

    # Split into train/val
    val_ratio = args.val_ratio
    val_size = max(1, int(len(samples) * val_ratio))
    train_samples = samples[:-val_size]
    val_samples = samples[-val_size:]

    print(f"Split: {len(train_samples)} train, {len(val_samples)} val")

    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")

    _write_jsonl(train_path, train_samples)
    _write_jsonl(val_path, val_samples)

    print(f"Wrote training data to {train_path}")
    print(f"Wrote validation data to {val_path}")

    # Compute statistics
    solved_count = sum(s["outcome_solved"] for s in samples)
    stats = {
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "solved_count": solved_count,
        "solved_rate": solved_count / max(1, len(samples)),
    }

    stats_path = os.path.join(args.output_dir, "dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Wrote statistics to {stats_path}")
    print(f"Solved rate: {stats['solved_rate']:.2%}")

    return {
        "train_path": train_path,
        "val_path": val_path,
        "stats_path": stats_path,
        "stats": stats,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Transformer router training dataset")
    parser.add_argument("--phase1_jsonl", required=True, help="Phase1 log file (JSONL)")
    parser.add_argument("--phase2_jsonl", required=True, help="Phase2 log file (JSONL)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation split ratio")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = build_dataset(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
