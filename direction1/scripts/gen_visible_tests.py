"""
Generate benchmarks/humaneval_visible_tests.jsonl from HumanEval prompt doctests.

Usage:
    python scripts/gen_visible_tests.py
"""
import json
import os
import re
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_jsonl


DOCTEST_RE = re.compile(r"^\s*>>>\s*(.+?)\s*$")


def extract_doctest_asserts(prompt: str) -> List[str]:
    lines = prompt.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        m = DOCTEST_RE.match(lines[i])
        if not m:
            i += 1
            continue
        expr = m.group(1).strip()
        j = i + 1
        expected_parts: List[str] = []
        while j < len(lines):
            nxt = lines[j].strip()
            if not nxt:
                break
            if nxt.startswith(">>>"):
                break
            if nxt.startswith("..."):
                break
            expected_parts.append(nxt)
            break
        if expected_parts:
            expected = expected_parts[0]
            out.append(f"assert {expr} == {expected}")
            i = j + 1
        else:
            i += 1
    return out


def main() -> None:
    src = "benchmarks/humaneval_full.jsonl"
    dst = "benchmarks/humaneval_visible_tests.jsonl"
    data = read_jsonl(src)
    rows = []
    for item in data:
        rows.append(
            {
                "task_id": item["task_id"],
                "entry_point": item["entry_point"],
                "given_tests": extract_doctest_asserts(item["prompt"]),
            }
        )

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    with_tests = sum(1 for r in rows if r["given_tests"])
    print(f"Generated {dst}: total={total}, with_tests={with_tests}")


if __name__ == "__main__":
    main()
