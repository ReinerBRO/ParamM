#!/usr/bin/env python3
"""Quality report for ParamMem synthesis dataset."""

from __future__ import annotations

import argparse
import json
import statistics
from typing import Dict, List


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    total = len(rows)

    src_counts: Dict[str, int] = {}
    pitfalls_len = []
    buggy_len = []
    doc_lens = []
    bad_func = 0

    for r in rows:
        src = str(r.get("source", "unknown"))
        src_counts[src] = src_counts.get(src, 0) + 1
        pitfalls = r.get("pitfalls", [])
        buggy = r.get("buggy_implementations", [])
        doc = str(r.get("docstring", ""))
        func = str(r.get("func_sign", ""))
        pitfalls_len.append(len(pitfalls) if isinstance(pitfalls, list) else 0)
        buggy_len.append(len(buggy) if isinstance(buggy, list) else 0)
        doc_lens.append(len(doc))
        if not func.startswith("def "):
            bad_func += 1

    report = {
        "total": total,
        "source_counts": src_counts,
        "pitfalls_len_mean": round(statistics.mean(pitfalls_len), 3) if pitfalls_len else 0,
        "pitfalls_len_min": min(pitfalls_len) if pitfalls_len else 0,
        "pitfalls_len_max": max(pitfalls_len) if pitfalls_len else 0,
        "buggy_len_mean": round(statistics.mean(buggy_len), 3) if buggy_len else 0,
        "buggy_len_min": min(buggy_len) if buggy_len else 0,
        "buggy_len_max": max(buggy_len) if buggy_len else 0,
        "doc_len_mean": round(statistics.mean(doc_lens), 3) if doc_lens else 0,
        "doc_len_min": min(doc_lens) if doc_lens else 0,
        "doc_len_max": max(doc_lens) if doc_lens else 0,
        "bad_func_sign_count": bad_func,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
