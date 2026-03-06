"""Build offline router training datasets for HumanEval and MBPP."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from memory_router.feature_schema import FEATURE_ORDER, normalize_mix, vectorize_features


MEMORY_TYPES = [
    "prompt_sim_positive",
    "reflection_sim_positive",
    "negative_trajectory_suppression",
]

def _safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "solved", "pass", "passed"}
    return False


def _read_records(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []

    records: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, Mapping):
                records.append(dict(item))
        return records

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return records

    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping):
                records.append(dict(item))
    elif isinstance(data, Mapping):
        if isinstance(data.get("records"), list):
            for item in data.get("records", []):
                if isinstance(item, Mapping):
                    records.append(dict(item))
        else:
            records.append(dict(data))
    return records


def _record_id(record: Mapping[str, Any]) -> str:
    for key in (
        "task_id",
        "id",
        "problem_id",
        "question_id",
        "name",
        "slug",
        "entry_point",
    ):
        if key in record and record[key] is not None:
            return str(record[key])
    prompt = str(record.get("prompt") or record.get("question") or "")
    if prompt:
        return prompt[:128]
    import hashlib

    payload = json.dumps(dict(record), sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"unknown_{digest}"


def _candidate_ids(record: Mapping[str, Any]) -> List[str]:
    ids: List[str] = []
    for key in (
        "task_id",
        "id",
        "problem_id",
        "question_id",
        "name",
        "slug",
        "entry_point",
    ):
        value = record.get(key)
        if value is not None:
            ids.append(str(value))
    prompt = record.get("prompt") or record.get("question")
    if prompt:
        ids.append(str(prompt)[:128])
    rid = _record_id(record)
    if rid:
        ids.append(rid)

    # Keep order while removing duplicates.
    seen = set()
    out: List[str] = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _extract_sims(source: Mapping[str, Any], keys: Iterable[str]) -> List[float]:
    values: List[float] = []
    for key in keys:
        val = source.get(key)
        if isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, Mapping):
                    for k in ("score", "sim", "similarity", "value"):
                        if k in item:
                            values.append(_to_float(item.get(k), 0.0))
                            break
                else:
                    values.append(_to_float(item, 0.0))
        elif isinstance(val, Mapping):
            for k in ("score", "sim", "similarity", "value"):
                if k in val:
                    values.append(_to_float(val.get(k), 0.0))
                    break
        elif val is not None:
            values.append(_to_float(val, 0.0))
    return [max(0.0, v) for v in values]


def _trajectory_to_record(traj: Mapping[str, Any], origin: str, traj_kind: str) -> Dict[str, Any]:
    rec = dict(traj)
    rec["_origin"] = origin
    rec["_traj_kind"] = traj_kind
    rec["_id"] = _record_id(rec)
    return rec


def _load_memory_records(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []

    if path.endswith(".pkl"):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            return []

        records: List[Dict[str, Any]] = []
        if isinstance(data, Mapping):
            pos = data.get("positive_trajectories")
            neg = data.get("negative_trajectories")
            if isinstance(pos, list):
                for traj in pos:
                    if isinstance(traj, Mapping):
                        records.append(_trajectory_to_record(traj, path, "positive"))
            if isinstance(neg, list):
                for traj in neg:
                    if isinstance(traj, Mapping):
                        records.append(_trajectory_to_record(traj, path, "negative"))
            if not records:
                records.append(dict(data))
            return records

        if isinstance(data, list):
            for item in data:
                if isinstance(item, Mapping):
                    records.append(dict(item))
            return records

        return []

    return _read_records(path)


def _first_non_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _build_primary_map(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    primary: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        primary[_record_id(rec)] = rec
    return primary


def _outcome_solved(phase1: Mapping[str, Any], phase2: Mapping[str, Any]) -> int:
    candidates = [
        phase2.get("solved"),
        phase2.get("pass"),
        phase2.get("passed"),
        phase2.get("is_solved"),
        phase2.get("success"),
        phase1.get("solved"),
        phase1.get("pass"),
        phase1.get("passed"),
        phase1.get("is_solved"),
        phase1.get("success"),
    ]
    for c in candidates:
        if c is not None:
            return int(_to_bool(c))
    return 0


def _build_state(task_id: str, phase1: Mapping[str, Any], phase2: Mapping[str, Any], mem: Mapping[str, Any]) -> Dict[str, Any]:
    state: Dict[str, Any] = {"task_id": task_id}
    state.update(_safe_dict(phase1))
    state.update(_safe_dict(phase2))

    state["phase1"] = _safe_dict(phase1)
    state["phase2"] = _safe_dict(phase2)
    state["memory"] = _safe_dict(mem)

    if "prompt" not in state:
        state["prompt"] = _first_non_none(phase2.get("prompt"), phase1.get("prompt"), mem.get("prompt"), "")
    if "reflections" not in state:
        state["reflections"] = _safe_list(
            _first_non_none(phase2.get("reflections"), phase1.get("reflections"), mem.get("reflections"), [])
        )
    if "errors" not in state:
        state["errors"] = _safe_list(
            _first_non_none(phase2.get("errors"), phase1.get("errors"), phase2.get("error_messages"), [])
        )
    return state


def _heuristic_mix(state: Mapping[str, Any], phase1: Mapping[str, Any], phase2: Mapping[str, Any], mem: Mapping[str, Any], solved: int) -> List[float]:
    mem_dict = _safe_dict(mem)
    p_vals = _extract_sims(
        mem_dict,
        ["prompt_sims", "prompt_sim", "prompt_similarity", "query_sims", "query_similarity"],
    )
    r_vals = _extract_sims(
        mem_dict,
        ["reflection_sims", "reflection_sim", "reflection_similarity", "analysis_sims"],
    )
    n_vals = _extract_sims(
        mem_dict,
        ["negative_sims", "negative_sim", "negative_similarity", "bad_sims", "failure_sims"],
    )

    # Pull from phase logs when memory logs miss fields.
    if not p_vals:
        p_vals = _extract_sims(_safe_dict(phase2), ["prompt_sim", "query_sim", "similarity"]) + _extract_sims(
            _safe_dict(phase1), ["prompt_sim", "query_sim", "similarity"]
        )
    if not r_vals:
        r_vals = _extract_sims(_safe_dict(phase2), ["reflection_sim", "analysis_sim"]) + _extract_sims(
            _safe_dict(phase1), ["reflection_sim", "analysis_sim"]
        )
    if not n_vals:
        n_vals = _extract_sims(_safe_dict(phase2), ["negative_sim", "failure_sim"]) + _extract_sims(
            _safe_dict(phase1), ["negative_sim", "failure_sim"]
        )

    prompt_signal = max(p_vals) if p_vals else 0.0
    reflection_signal = max(r_vals) if r_vals else 0.0
    negative_signal = max(n_vals) if n_vals else 0.0

    errors = _safe_list(state.get("errors")) + _safe_list(state.get("failures"))
    errors_text = " ".join(str(e).lower() for e in errors)
    failure_boost = 0.0
    for key, inc in (("syntax", 0.10), ("assert", 0.12), ("timeout", 0.16), ("traceback", 0.10), ("exception", 0.08)):
        if key in errors_text:
            failure_boost += inc

    reflection_rounds = len(_safe_list(state.get("reflections")))

    w_prompt = max(0.0, 0.30 + prompt_signal)
    w_reflection = max(0.0, 0.25 + reflection_signal + 0.06 * reflection_rounds)
    w_negative = max(0.0, 0.20 + negative_signal + failure_boost)

    if solved:
        w_prompt += 0.25
        w_reflection += 0.20
    else:
        w_negative += 0.35

    return normalize_mix([w_prompt, w_reflection, w_negative])


def _build_memory_stats(
    mem_records: List[Dict[str, Any]],
    task_id: str,
    phase1: Mapping[str, Any],
    phase2: Mapping[str, Any],
) -> Dict[str, Any]:
    target_ids = {task_id}
    target_ids.update(_candidate_ids(_safe_dict(phase1)))
    target_ids.update(_candidate_ids(_safe_dict(phase2)))

    matched = [
        rec for rec in mem_records
        if set(_candidate_ids(rec)) & target_ids
    ]

    prompt_vals: List[float] = []
    reflection_vals: List[float] = []
    negative_vals: List[float] = []
    retrieval_candidates = 0

    for rec in matched:
        kind = str(rec.get("_traj_kind", "")).lower()
        p = _extract_sims(rec, ["prompt_sim", "prompt_similarity", "similarity", "score"])
        r = _extract_sims(rec, ["reflection_sim", "reflection_similarity", "similarity", "score"])
        prompt_vals.extend(p)
        reflection_vals.extend(r)

        if kind == "negative":
            n = _extract_sims(rec, ["negative_sim", "similarity", "score"])
            if not n:
                n = [max(p + r + [0.0])]
            negative_vals.extend(n)

        retrieval_candidates += 1

    if not negative_vals:
        negative_vals = _extract_sims(
            {"negative_sims": [rec.get("similarity") for rec in matched]},
            ["negative_sims"],
        )

    return {
        "prompt_sims": prompt_vals,
        "reflection_sims": reflection_vals,
        "negative_sims": negative_vals,
        "retrieval_candidates": retrieval_candidates,
        "matched_memory_records": len(matched),
    }


def _build_samples(phase1_path: str, phase2_path: str, mem_path: str, dataset_name: str) -> List[Dict[str, Any]]:
    p1_records = _read_records(phase1_path)
    p2_records = _read_records(phase2_path)
    mem_records = _load_memory_records(mem_path)

    p1_map = _build_primary_map(p1_records)
    p2_map = _build_primary_map(p2_records)

    all_ids = sorted(set(p1_map.keys()) | set(p2_map.keys()))

    samples: List[Dict[str, Any]] = []
    for task_id in all_ids:
        phase1 = _safe_dict(p1_map.get(task_id, {}))
        phase2 = _safe_dict(p2_map.get(task_id, {}))
        mem_stats = _build_memory_stats(mem_records, task_id, phase1, phase2)

        state = _build_state(task_id, phase1, phase2, mem_stats)
        solved = _outcome_solved(phase1, phase2)
        mix = _heuristic_mix(state, phase1, phase2, mem_stats, solved)
        features = vectorize_features(state, FEATURE_ORDER)

        sample = {
            "dataset": dataset_name,
            "task_id": task_id,
            "state": state,
            "feature_order": FEATURE_ORDER,
            "features": features,
            "target_mix": mix,
            "outcome_solved": int(solved),
            "memory_types": MEMORY_TYPES,
        }
        samples.append(sample)

    return samples


def _write_jsonl(path: str, records: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _dataset_stats(samples: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not samples:
        return {
            "sample_count": 0,
            "solved_count": 0,
            "solved_rate": 0.0,
            "avg_mix": [0.0, 0.0, 0.0],
        }

    solved = sum(int(s.get("outcome_solved", 0)) for s in samples)
    mix_sum = [0.0, 0.0, 0.0]
    for s in samples:
        mix = s.get("target_mix") or [0.0, 0.0, 0.0]
        for i in range(3):
            mix_sum[i] += _to_float(mix[i] if i < len(mix) else 0.0)

    n = len(samples)
    return {
        "sample_count": n,
        "solved_count": solved,
        "solved_rate": solved / max(1, n),
        "avg_mix": [x / n for x in mix_sum],
    }


def _split_task_ids(task_ids: List[str], val_ratio: float = 0.2) -> Tuple[List[str], List[str]]:
    ordered = sorted(dict.fromkeys(task_ids))
    if not ordered:
        return [], []
    if len(ordered) == 1:
        return ordered, []
    val_n = max(1, int(round(len(ordered) * val_ratio)))
    val_n = min(val_n, len(ordered) - 1)
    train_ids = ordered[:-val_n]
    val_ids = ordered[-val_n:]
    return train_ids, val_ids


def build_datasets(args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)

    humaneval_samples = _build_samples(args.humaneval_phase1, args.humaneval_phase2, args.humaneval_mem, "humaneval")
    mbpp_samples = _build_samples(args.mbpp_phase1, args.mbpp_phase2, args.mbpp_mem, "mbpp")

    train_h_path = os.path.join(args.output_dir, "train_humaneval.jsonl")
    train_m_path = os.path.join(args.output_dir, "train_mbpp.jsonl")
    split_path = os.path.join(args.output_dir, "cross_eval_splits.json")
    stats_path = os.path.join(args.output_dir, "dataset_stats.json")

    _write_jsonl(train_h_path, humaneval_samples)
    _write_jsonl(train_m_path, mbpp_samples)

    train_h_ids, val_h_ids = _split_task_ids([str(s.get("task_id", "")) for s in humaneval_samples])
    train_m_ids, val_m_ids = _split_task_ids([str(s.get("task_id", "")) for s in mbpp_samples])

    splits = {
        "train_humaneval": train_h_path,
        "train_mbpp": train_m_path,
        "cross_eval": {
            "humaneval_to_mbpp": {
                "train": train_h_path,
                "eval": train_m_path,
            },
            "mbpp_to_humaneval": {
                "train": train_m_path,
                "eval": train_h_path,
            },
        },
        "intra_dataset": {
            "humaneval": {
                "train_task_ids": train_h_ids,
                "val_task_ids": val_h_ids,
            },
            "mbpp": {
                "train_task_ids": train_m_ids,
                "val_task_ids": val_m_ids,
            },
        },
        "feature_order": FEATURE_ORDER,
        "notes": "Task IDs are disjoint between train/val splits to avoid leakage.",
    }

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    stats = {
        "humaneval": _dataset_stats(humaneval_samples),
        "mbpp": _dataset_stats(mbpp_samples),
        "total_samples": len(humaneval_samples) + len(mbpp_samples),
        "memory_types": MEMORY_TYPES,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return {
        "train_humaneval": train_h_path,
        "train_mbpp": train_m_path,
        "cross_eval_splits": split_path,
        "dataset_stats": stats_path,
        "stats": stats,
    }



def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build direction3 router datasets from phase logs.")
    parser.add_argument("--humaneval_phase1", required=True, help="Path to HumanEval phase1 log (json/jsonl)")
    parser.add_argument("--humaneval_phase2", required=True, help="Path to HumanEval phase2 log (json/jsonl)")
    parser.add_argument("--humaneval_mem", required=True, help="Path to HumanEval memory log (json/jsonl)")
    parser.add_argument("--mbpp_phase1", required=True, help="Path to MBPP phase1 log (json/jsonl)")
    parser.add_argument("--mbpp_phase2", required=True, help="Path to MBPP phase2 log (json/jsonl)")
    parser.add_argument("--mbpp_mem", required=True, help="Path to MBPP memory log (json/jsonl)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = build_datasets(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
