#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <train_jsonl> <val_jsonl> <output_ckpt_or_dir> [epochs] [batch_size] [lr] [seed]"
  exit 1
fi

PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m memory_router.train_router \
  --train_jsonl "$1" \
  --val_jsonl "$2" \
  --output_ckpt "$3" \
  --epochs "${4:-20}" \
  --batch_size "${5:-32}" \
  --lr "${6:-0.001}" \
  --seed "${7:-42}"

echo "Router training done. Output target: $3"
