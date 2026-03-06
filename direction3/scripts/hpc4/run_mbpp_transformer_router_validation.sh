#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/user/user06/ParamAgent/direction3"
RUN_NAME="${1:-dir3_router_mbpp_transformer_val_$(date +%Y%m%d_%H%M%S)}"

export ROUTER_CKPT="${ROUTER_CKPT:-${PROJECT_ROOT}/router_checkpoints/transformer_mbpp_v1/router.ckpt}"
export ROUTER_CONF_THRESHOLD="${ROUTER_CONF_THRESHOLD:-0.6}"
export NUM_WORKERS="${NUM_WORKERS:-24}"

echo "[VALIDATION] run_name=${RUN_NAME}"
echo "[VALIDATION] router_ckpt=${ROUTER_CKPT}"
echo "[VALIDATION] router_conf_threshold=${ROUTER_CONF_THRESHOLD}"
echo "[VALIDATION] num_workers=${NUM_WORKERS}"

bash "${PROJECT_ROOT}/scripts/hpc4/run_mbpp_router_fixed_pipeline.sh" "${RUN_NAME}"

RUN_ROOT="${PROJECT_ROOT}/results/mbpp_router_runs/${RUN_NAME}"
if [[ -f "${RUN_ROOT}/results.txt" ]]; then
  echo "[VALIDATION] results: ${RUN_ROOT}/results.txt"
  cat "${RUN_ROOT}/results.txt"
fi
