#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: run_direction1_acceptance.sh <linear_run_dir> <search_run_dir> <dataset_name>"
  exit 1
fi

LINEAR_RUN_DIR="$1"
SEARCH_RUN_DIR="$2"
DATASET_NAME="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
REPORT_DIR="$ROOT_DIR/results/ab_reports"

mkdir -p "$REPORT_DIR"

LINEAR_RUN_NAME="$(basename "$LINEAR_RUN_DIR")"
SEARCH_RUN_NAME="$(basename "$SEARCH_RUN_DIR")"

LINEAR_REPORT_JSON="$REPORT_DIR/${LINEAR_RUN_NAME}.json"
SEARCH_REPORT_JSON="$REPORT_DIR/${SEARCH_RUN_NAME}.json"

python "$SCRIPT_DIR/eval_direction1_ab.py" "$LINEAR_RUN_DIR" --output "$LINEAR_REPORT_JSON"
python "$SCRIPT_DIR/eval_direction1_ab.py" "$SEARCH_RUN_DIR" --output "$SEARCH_REPORT_JSON"

python "$SCRIPT_DIR/compare_against_baseline.py" \
  "$LINEAR_REPORT_JSON" \
  "$SEARCH_REPORT_JSON" \
  "$DATASET_NAME" \
  --docs_baseline "$ROOT_DIR/docs/实验结果数据.md" \
  --output "$REPORT_DIR/acceptance_${DATASET_NAME}.json"

echo "Acceptance report: $REPORT_DIR/acceptance_${DATASET_NAME}.json"
