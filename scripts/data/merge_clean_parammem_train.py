#!/usr/bin/env python3
"""Merge APPS + synth ParamMem supervision and normalize to training-ready JSONL."""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


REQUIRED_KEYS = ["func_sign", "docstring", "pitfalls", "buggy_implementations"]


@dataclass
class CleanResult:
    record: dict | None
    reason: str
    bug_total: int
    bug_kept: int
    bug_non_str_or_bad: int
    bug_ast_bad: int


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _norm_str_list(values: object) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for v in values:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _normalize_buggy(values: object) -> Tuple[List[str], int, int, int]:
    if not isinstance(values, list):
        return [], 0, 0, 0

    out: List[str] = []
    seen = set()
    bug_non_str_or_bad = 0
    bug_ast_bad = 0

    for item in values:
        code = None
        if isinstance(item, str):
            code = item.strip()
        elif isinstance(item, dict):
            maybe = item.get("code")
            if isinstance(maybe, str):
                code = maybe.strip()
            else:
                bug_non_str_or_bad += 1
        else:
            bug_non_str_or_bad += 1

        if not code:
            if item is not None and not isinstance(item, (str, dict)):
                bug_non_str_or_bad += 0
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(code)
        except SyntaxError:
            bug_ast_bad += 1
            continue

        if code in seen:
            continue
        seen.add(code)
        out.append(code)

    return out, len(values), bug_non_str_or_bad, bug_ast_bad


def clean_record(row: dict, min_pitfalls: int, min_buggy: int) -> CleanResult:
    for k in REQUIRED_KEYS:
        if k not in row:
            return CleanResult(None, f"missing_{k}", 0, 0, 0, 0)

    func_sign = row.get("func_sign")
    if not isinstance(func_sign, str) or not func_sign.strip().startswith("def "):
        return CleanResult(None, "bad_func_sign", 0, 0, 0, 0)

    docstring = row.get("docstring")
    if not isinstance(docstring, str) or len(docstring.strip()) < 20:
        return CleanResult(None, "bad_docstring", 0, 0, 0, 0)

    pitfalls = _norm_str_list(row.get("pitfalls"))
    if len(pitfalls) < min_pitfalls:
        return CleanResult(None, "bad_pitfalls", 0, 0, 0, 0)

    buggy, bug_total, bug_non_str_or_bad, bug_ast_bad = _normalize_buggy(row.get("buggy_implementations"))
    if len(buggy) < min_buggy:
        return CleanResult(
            None,
            "bad_buggy_implementations",
            bug_total,
            len(buggy),
            bug_non_str_or_bad,
            bug_ast_bad,
        )

    cleaned = dict(row)
    cleaned["func_sign"] = func_sign.strip()
    cleaned["docstring"] = docstring.strip()
    cleaned["pitfalls"] = pitfalls
    cleaned["buggy_implementations"] = buggy

    return CleanResult(
        cleaned,
        "ok",
        bug_total,
        len(buggy),
        bug_non_str_or_bad,
        bug_ast_bad,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apps", required=True)
    parser.add_argument("--synth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min_pitfalls", type=int, default=4)
    parser.add_argument("--min_buggy", type=int, default=4)
    args = parser.parse_args()

    apps_path = Path(args.apps)
    synth_path = Path(args.synth)
    out_path = Path(args.output)
    report_path = Path(args.report)

    apps_rows = read_jsonl(apps_path)
    synth_rows = [r for r in read_jsonl(synth_path) if str(r.get("source")) == "synth"]

    merged_input = apps_rows + synth_rows

    dropped_reasons: Dict[str, int] = {}
    source_in: Dict[str, int] = {}
    source_out: Dict[str, int] = {}
    total_bug_before = 0
    total_bug_after = 0
    total_bug_non_str_or_bad = 0
    total_bug_ast_bad = 0

    dedup: Dict[str, dict] = {}
    for row in merged_input:
        source = str(row.get("source", "unknown"))
        source_in[source] = source_in.get(source, 0) + 1

        res = clean_record(row, min_pitfalls=args.min_pitfalls, min_buggy=args.min_buggy)
        total_bug_before += res.bug_total
        total_bug_after += res.bug_kept
        total_bug_non_str_or_bad += res.bug_non_str_or_bad
        total_bug_ast_bad += res.bug_ast_bad

        if res.record is None:
            dropped_reasons[res.reason] = dropped_reasons.get(res.reason, 0) + 1
            continue

        sid = str(res.record.get("sample_id", ""))
        if not sid:
            dropped_reasons["missing_sample_id"] = dropped_reasons.get("missing_sample_id", 0) + 1
            continue
        dedup[sid] = res.record

    rows = list(dedup.values())
    rows.sort(key=lambda x: str(x.get("sample_id", "")))

    for r in rows:
        source = str(r.get("source", "unknown"))
        source_out[source] = source_out.get(source, 0) + 1

    write_jsonl(out_path, rows)

    report = {
        "apps_path": str(apps_path),
        "synth_path": str(synth_path),
        "synth_filter": "source==synth",
        "input_total": len(merged_input),
        "input_source_counts": source_in,
        "output_total": len(rows),
        "output_source_counts": source_out,
        "dropped_total": len(merged_input) - len(rows),
        "dropped_reasons": dropped_reasons,
        "buggy_stats": {
            "total_before_clean": total_bug_before,
            "total_after_clean": total_bug_after,
            "removed_non_string_or_bad_dict": total_bug_non_str_or_bad,
            "removed_ast_syntax_error": total_bug_ast_bad,
        },
        "constraints": {
            "min_pitfalls": args.min_pitfalls,
            "min_buggy": args.min_buggy,
        },
        "output_path": str(out_path),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
