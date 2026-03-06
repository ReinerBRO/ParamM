#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_PATH="${1:-benchmarks/humaneval-py.jsonl}"
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
RUN_NAME_BASE="${RUN_NAME_BASE:-humaneval_baseline}"
RUN_NAME_ROUTER="${RUN_NAME_ROUTER:-humaneval_router_on}"
ROUTER_CKPT="${ROUTER_CKPT:-$ROOT_DIR/results/router_train/router.ckpt}"

BASE_OUT="$ROOT_DIR/results/router_eval/${RUN_NAME_BASE}"
ROUTER_OUT="$ROOT_DIR/results/router_eval/${RUN_NAME_ROUTER}"

mkdir -p "$BASE_OUT" "$ROUTER_OUT"

# Baseline template (router OFF)
"$PYTHON_BIN" "$ROOT_DIR/main_param.py" \
  --run_name "$RUN_NAME_BASE" \
  --root_dir "$BASE_OUT" \
  --dataset_path "$DATASET_PATH" \
  --strategy dot \
  --language py \
  --model "$MODEL_NAME" \
  --max_iters 3 \
  --inner_iter 5

# Router template (router ON)
"$PYTHON_BIN" "$ROOT_DIR/main_param.py" \
  --run_name "$RUN_NAME_ROUTER" \
  --root_dir "$ROUTER_OUT" \
  --dataset_path "$DATASET_PATH" \
  --strategy dot \
  --language py \
  --model "$MODEL_NAME" \
  --max_iters 3 \
  --inner_iter 5 \
  --router_enable \
  --router_conf_threshold "${ROUTER_CONF_THRESHOLD:-0.6}" \
  --router_ckpt_path "$ROUTER_CKPT"
