#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple


NOISE_PATTERNS = [
    r"<<SYS>>",
    r"<</SYS>>",
    r"\[/?INST\]",
    r"FUNC_SIGNATURE:",
    r"Now, list potential pitfalls.*",
]


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    out = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    for pat in NOISE_PATTERNS:
        out = re.sub(pat, "", out, flags=re.IGNORECASE | re.DOTALL)
    return out.strip()


def extract_numbered_pitfalls(text: str) -> Tuple[str, Dict[str, int]]:
    clean = normalize_text(text)
    lines = [ln.rstrip() for ln in clean.split("\n")]
    bullets: List[str] = []
    cur: List[str] = []
    meta = {"had_numbered": 0, "fallback_used": 0}

    bullet_re = re.compile(r"^\s*(\d+)[\).]\s*(.*)$")
    for line in lines:
        m = bullet_re.match(line)
        if m:
            if cur:
                bullets.append(" ".join(x.strip() for x in cur if x.strip()))
            cur = [m.group(2).strip()]
            continue
        if cur:
            if line.strip():
                cur.append(line.strip())
        elif line.strip().startswith("-"):
            cur = [line.strip().lstrip("-").strip()]
    if cur:
        bullets.append(" ".join(x.strip() for x in cur if x.strip()))

    bullets = [b for b in bullets if b]
    if bullets:
        meta["had_numbered"] = 1
        return "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets)), meta

    meta["fallback_used"] = 1
    s = clean
    s = re.sub(r"^\s*\[?Pitfalls?\]?:?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if not s:
        s = "1. Check edge cases and boundary conditions."
    return s, meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--report_json", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl)
    out: List[dict] = []
    stats = {
        "input_rows": len(rows),
        "numbered_pitfall_rows": 0,
        "numbered_high_temp_rows": 0,
        "fallback_pitfall_rows": 0,
        "fallback_high_temp_rows": 0,
    }

    for row in rows:
        rec = dict(row)
        pit, m1 = extract_numbered_pitfalls(str(rec.get("pitfall", "")))
        hpit, m2 = extract_numbered_pitfalls(str(rec.get("high_temp_pitfall", "")))
        rec["pitfall"] = pit
        rec["high_temp_pitfall"] = hpit
        out.append(rec)

        stats["numbered_pitfall_rows"] += m1["had_numbered"]
        stats["numbered_high_temp_rows"] += m2["had_numbered"]
        stats["fallback_pitfall_rows"] += m1["fallback_used"]
        stats["fallback_high_temp_rows"] += m2["fallback_used"]

    write_jsonl(args.output_jsonl, out)
    os.makedirs(os.path.dirname(args.report_json) or ".", exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
