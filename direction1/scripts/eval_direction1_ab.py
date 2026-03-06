#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple


REQUIRED_SEARCH_FIELDS = [
    "search_nodes_expanded",
    "verifier_cost",
    "best_score_trace",
]


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _pass_rate(rows: List[Dict[str, Any]]) -> Tuple[int, int, Optional[float]]:
    total = len(rows)
    if total == 0:
        return 0, 0, None
    solved = sum(1 for row in rows if bool(row.get("is_solved", False)))
    return solved, total, solved / total


def _coverage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        result: Dict[str, Any] = {
            "row_count": 0,
            "all_required_fields": 0.0,
        }
        for field in REQUIRED_SEARCH_FIELDS:
            result[field] = 0.0
        return result

    per_field: Dict[str, float] = {}
    for field in REQUIRED_SEARCH_FIELDS:
        covered = sum(1 for row in rows if (field in row and row.get(field) is not None))
        per_field[field] = covered / total

    covered_all = sum(
        1
        for row in rows
        if all((field in row and row.get(field) is not None) for field in REQUIRED_SEARCH_FIELDS)
    )

    result = {
        "row_count": total,
        "all_required_fields": covered_all / total,
        **per_field,
    }
    return result


def _mode_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        mode = row.get("phase2_search_mode", "unknown")
        if not isinstance(mode, str):
            mode = str(mode)
        counts[mode] = counts.get(mode, 0) + 1
    return counts


def _default_output_path(run_dir: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(root_dir, "results", "ab_reports")
    run_name = os.path.basename(os.path.normpath(run_dir))
    return os.path.join(output_dir, f"{run_name}.json")


def build_report(run_dir: str) -> Dict[str, Any]:
    run_dir_abs = os.path.abspath(run_dir)
    run_name = os.path.basename(os.path.normpath(run_dir_abs))

    first_stage_path = os.path.join(run_dir_abs, "first_stage_log.jsonl")
    second_stage_path = os.path.join(run_dir_abs, "second_stage_log.jsonl")

    if not os.path.exists(first_stage_path):
        raise FileNotFoundError(f"Missing first stage log: {first_stage_path}")

    first_rows = _read_jsonl(first_stage_path)
    second_rows = _read_jsonl(second_stage_path) if os.path.exists(second_stage_path) else []

    phase1_solved, phase1_total, phase1_pass_rate = _pass_rate(first_rows)
    phase2_solved, phase2_total, phase2_pass_rate = _pass_rate(second_rows)

    metric_rows = second_rows if second_rows else first_rows
    metric_source = "second_stage_log.jsonl" if second_rows else "first_stage_log.jsonl"

    total_cost = sum(_to_float(row.get("cost", 0.0)) for row in metric_rows)
    prompt_tokens = sum(_to_int(row.get("prompt_tokens", 0)) for row in metric_rows)
    completion_tokens = sum(_to_int(row.get("completion_tokens", 0)) for row in metric_rows)
    total_tokens = prompt_tokens + completion_tokens

    coverage = _coverage(metric_rows)
    mode_count = _mode_counts(metric_rows)
    fallback_count = sum(1 for row in metric_rows if bool(row.get("search_fallback_to_linear", False)))

    report = {
        "run_name": run_name,
        "run_dir": run_dir_abs,
        "first_stage_log": first_stage_path,
        "second_stage_log": second_stage_path if os.path.exists(second_stage_path) else None,
        "phase1_solved": phase1_solved,
        "phase1_total": phase1_total,
        "phase1_pass_rate": phase1_pass_rate,
        "phase2_solved": phase2_solved,
        "phase2_total": phase2_total,
        "phase2_pass_rate": phase2_pass_rate,
        "metrics_source": metric_source,
        "total_cost": total_cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "search_fields_coverage": coverage,
        "search_mode_count": mode_count,
        "fallback_count": fallback_count,
    }
    return report


def _fmt_rate(rate: Optional[float]) -> str:
    if rate is None:
        return "NA"
    return f"{rate * 100.0:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Direction1 A/B metrics report from first/second stage logs."
    )
    parser.add_argument("run_dir", help="Run directory that contains first_stage_log.jsonl and second_stage_log.jsonl")
    parser.add_argument("--output", default=None, help="Output JSON path; default is direction1/results/ab_reports/<run_name>.json")
    args = parser.parse_args()

    report = build_report(args.run_dir)
    output_path = os.path.abspath(args.output) if args.output else _default_output_path(args.run_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True)

    coverage_all = report["search_fields_coverage"]["all_required_fields"]
    print(
        f"run={report['run_name']} phase1={_fmt_rate(report['phase1_pass_rate'])} "
        f"phase2={_fmt_rate(report['phase2_pass_rate'])} "
        f"cost=${report['total_cost']:.6f} tokens={report['total_tokens']}"
    )
    print(
        f"search_mode_count={report['search_mode_count']} "
        f"fallback_count={report['fallback_count']} "
        f"coverage_all={coverage_all * 100.0:.2f}% "
        f"report={output_path}"
    )


if __name__ == "__main__":
    main()
