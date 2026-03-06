#!/usr/bin/env python3
import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional


PASS_THRESHOLD_ABSOLUTE_POINTS = 1.5
TOKEN_GROWTH_THRESHOLD = 0.25
COST_GROWTH_THRESHOLD = 0.25


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return data


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dataset_field(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower()
    if normalized == "humaneval":
        return "HumanEval"
    if normalized == "mbpp":
        return "MBPP"
    raise ValueError("dataset_name must be one of: humaneval, mbpp")


def _extract_baseline_phase2_rate(docs_path: str, dataset_name: str) -> float:
    target_column = _dataset_field(dataset_name)
    with open(docs_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    row_match = re.search(
        r"<td>\s*ParamAgent-plus\s*\(Phase2\)\s*</td>(?P<rest>.*?)</tr>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if row_match is None:
        raise ValueError("Could not find ParamAgent-plus (Phase2) row in baseline docs")

    row_text = row_match.group("rest")
    values = re.findall(r"<td>\s*([^<]+?)\s*</td>", row_text, flags=re.IGNORECASE | re.DOTALL)
    if len(values) < 2:
        raise ValueError("Unexpected table format in baseline docs")

    # For this table, first two values are Llama-3.1-8B HumanEval/MBPP.
    value = values[0] if target_column == "HumanEval" else values[1]
    pct_match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", value)
    if pct_match is None:
        raise ValueError(f"Could not parse percentage from baseline value: {value}")
    return float(pct_match.group(1)) / 100.0


def _resolve_phase2_rate(report: Dict[str, Any]) -> Optional[float]:
    value = report.get("phase2_pass_rate")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio_delta(new_value: float, old_value: float) -> Optional[float]:
    if old_value <= 0:
        return None
    return (new_value - old_value) / old_value


def compare(
    docs_path: str,
    linear_report_path: str,
    search_report_path: str,
    dataset_name: str,
) -> Dict[str, Any]:
    baseline_phase2_rate = _extract_baseline_phase2_rate(docs_path, dataset_name)

    linear_report = _load_json(linear_report_path)
    search_report = _load_json(search_report_path)

    linear_phase2_rate = _resolve_phase2_rate(linear_report)
    search_phase2_rate = _resolve_phase2_rate(search_report)

    if linear_phase2_rate is None:
        raise ValueError("linear report missing valid phase2_pass_rate")
    if search_phase2_rate is None:
        raise ValueError("search report missing valid phase2_pass_rate")

    pass_gain_points = (search_phase2_rate - linear_phase2_rate) * 100.0

    linear_prompt_tokens = int(_to_float(linear_report.get("prompt_tokens", 0)))
    linear_completion_tokens = int(_to_float(linear_report.get("completion_tokens", 0)))
    linear_total_tokens = int(_to_float(linear_report.get("total_tokens", linear_prompt_tokens + linear_completion_tokens)))
    linear_total_cost = _to_float(linear_report.get("total_cost", 0.0))

    search_prompt_tokens = int(_to_float(search_report.get("prompt_tokens", 0)))
    search_completion_tokens = int(_to_float(search_report.get("completion_tokens", 0)))
    search_total_tokens = int(_to_float(search_report.get("total_tokens", search_prompt_tokens + search_completion_tokens)))
    search_total_cost = _to_float(search_report.get("total_cost", 0.0))

    token_growth_ratio = _ratio_delta(search_total_tokens, linear_total_tokens)
    cost_growth_ratio = _ratio_delta(search_total_cost, linear_total_cost)

    reasons: List[str] = []
    accept = True

    if pass_gain_points >= PASS_THRESHOLD_ABSOLUTE_POINTS:
        reasons.append(
            f"PASS: phase2 absolute gain {pass_gain_points:.2f} points >= {PASS_THRESHOLD_ABSOLUTE_POINTS:.2f}."
        )
    else:
        reasons.append(
            f"FAIL: phase2 absolute gain {pass_gain_points:.2f} points < {PASS_THRESHOLD_ABSOLUTE_POINTS:.2f}."
        )
        accept = False

    if token_growth_ratio is None:
        reasons.append("WARN: linear total_tokens is 0, cannot compute token growth ratio.")
    elif token_growth_ratio <= TOKEN_GROWTH_THRESHOLD:
        reasons.append(
            f"PASS: token growth {token_growth_ratio * 100.0:.2f}% <= {TOKEN_GROWTH_THRESHOLD * 100.0:.2f}%."
        )
    else:
        reasons.append(
            f"FAIL: token growth {token_growth_ratio * 100.0:.2f}% > {TOKEN_GROWTH_THRESHOLD * 100.0:.2f}%."
        )
        accept = False

    if cost_growth_ratio is None:
        reasons.append("WARN: linear total_cost is 0, cannot compute cost growth ratio.")
    elif cost_growth_ratio <= COST_GROWTH_THRESHOLD:
        reasons.append(
            f"PASS: cost growth {cost_growth_ratio * 100.0:.2f}% <= {COST_GROWTH_THRESHOLD * 100.0:.2f}%."
        )
    else:
        reasons.append(
            f"FAIL: cost growth {cost_growth_ratio * 100.0:.2f}% > {COST_GROWTH_THRESHOLD * 100.0:.2f}%."
        )
        accept = False

    search_cov = search_report.get("search_fields_coverage", {})
    coverage_all = _to_float(search_cov.get("all_required_fields", 0.0))
    if coverage_all >= 1.0:
        reasons.append("PASS: search log coverage has all required fields.")
    else:
        reasons.append(
            f"FAIL: search log coverage all_required_fields={coverage_all:.4f}, expected 1.0."
        )
        accept = False

    fallback_count = int(_to_float(search_report.get("fallback_count", 0)))
    if fallback_count > 0:
        reasons.append(f"INFO: search fallback triggered {fallback_count} times.")
    else:
        reasons.append("INFO: no search fallback observed.")

    result = {
        "dataset_name": dataset_name.lower(),
        "docs_baseline_path": os.path.abspath(docs_path),
        "baseline_paramagent_plus_phase2_pass_rate": baseline_phase2_rate,
        "linear_report_path": os.path.abspath(linear_report_path),
        "search_report_path": os.path.abspath(search_report_path),
        "linear_phase2_pass_rate": linear_phase2_rate,
        "search_phase2_pass_rate": search_phase2_rate,
        "search_vs_linear_absolute_gain_points": pass_gain_points,
        "search_vs_linear_token_growth_ratio": token_growth_ratio,
        "search_vs_linear_cost_growth_ratio": cost_growth_ratio,
        "acceptance_thresholds": {
            "phase2_absolute_gain_points": PASS_THRESHOLD_ABSOLUTE_POINTS,
            "token_growth_ratio_max": TOKEN_GROWTH_THRESHOLD,
            "cost_growth_ratio_max": COST_GROWTH_THRESHOLD,
            "search_fields_coverage_all_required": 1.0,
        },
        "accept": accept,
        "reasons": reasons,
    }
    return result


def _default_docs_basename() -> str:
    return "\u5b9e\u9a8c\u7ed3\u679c\u6570\u636e.md"


def _default_docs_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    return os.path.join(root_dir, "docs", _default_docs_basename())


def _default_output_path(dataset_name: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(root_dir, "results", "ab_reports")
    return os.path.join(output_dir, f"acceptance_{dataset_name.lower()}.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Direction1 search report against linear baseline and emit acceptance JSON."
    )
    parser.add_argument("linear_report_json", help="Path to linear mode report JSON")
    parser.add_argument("search_report_json", help="Path to search mode report JSON")
    parser.add_argument("dataset_name", help="Dataset name: humaneval or mbpp")
    parser.add_argument(
        "--docs_baseline",
        default=_default_docs_path(),
        help="Baseline docs file path (default: direction1/docs/实验结果数据.md)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: direction1/results/ab_reports/acceptance_<dataset>.json)",
    )
    args = parser.parse_args()

    result = compare(
        docs_path=args.docs_baseline,
        linear_report_path=args.linear_report_json,
        search_report_path=args.search_report_json,
        dataset_name=args.dataset_name,
    )

    output_path = os.path.abspath(args.output) if args.output else _default_output_path(args.dataset_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=True)

    print(
        f"dataset={result['dataset_name']} accept={result['accept']} "
        f"gain={result['search_vs_linear_absolute_gain_points']:.2f}pt "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
