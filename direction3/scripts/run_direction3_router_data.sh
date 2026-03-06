#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <humaneval_phase1> <humaneval_phase2> <humaneval_mem> <mbpp_phase1> <mbpp_phase2> <mbpp_mem> <output_dir>"
  exit 1
fi

PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m memory_router.dataset_builder \
  --humaneval_phase1 "$1" \
  --humaneval_phase2 "$2" \
  --humaneval_mem "$3" \
  --mbpp_phase1 "$4" \
  --mbpp_phase2 "$5" \
  --mbpp_mem "$6" \
  --output_dir "$7"

echo "Router data build done. Outputs in: $7"
