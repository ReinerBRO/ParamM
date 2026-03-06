#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?Usage: run_mbpp_phase12_pipeline_miku.sh <run_name> [env_name] [num_workers]}"
ENV_NAME="${2:-paramm}"
NUM_WORKERS="${3:-24}"
PROJECT_ROOT="/data/user/user06/ParamAgent"

RUN_ROOT="results/mbpp/paramAgent/${RUN_NAME}"
PIPELINE_LOG="${RUN_ROOT}/pipeline.log"

source /data/user/user06/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if [ ! -f "${RUN_ROOT}/worker_pids.tsv" ]; then
  echo "[ERROR] Missing phase1 pid file: ${RUN_ROOT}/worker_pids.tsv"
  exit 1
fi

echo "[PIPELINE] start $(date)" >> "${PIPELINE_LOG}"

while true; do
  alive=0
  while IFS=$'\t' read -r _ pid; do
    [ -n "${pid:-}" ] || continue
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done < "${RUN_ROOT}/worker_pids.tsv"
  echo "[PIPELINE] phase1_alive=${alive} $(date)" >> "${PIPELINE_LOG}"
  [ "${alive}" -eq 0 ] && break
  sleep 30
done

"${PYTHON_BIN}" scripts/hpc4/merge_phase1_shards.py \
  --run_root "${RUN_ROOT}" \
  --dataset_path benchmarks/mbpp-py.jsonl \
  --num_workers "${NUM_WORKERS}" \
  >> "${PIPELINE_LOG}" 2>&1

NUM_WORKERS="${NUM_WORKERS}" bash scripts/hpc4/run_mbpp_phase2_sharded_miku.sh "${ENV_NAME}" "${RUN_NAME}" >> "${PIPELINE_LOG}" 2>&1

while true; do
  alive=0
  while IFS=$'\t' read -r _ pid; do
    [ -n "${pid:-}" ] || continue
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done < "${RUN_ROOT}/phase2_worker_pids.tsv"
  echo "[PIPELINE] phase2_alive=${alive} $(date)" >> "${PIPELINE_LOG}"
  [ "${alive}" -eq 0 ] && break
  sleep 30
done

"${PYTHON_BIN}" scripts/hpc4/merge_phase2_shards.py \
  --run_root "${RUN_ROOT}" \
  --dataset_path benchmarks/mbpp-py.jsonl \
  --num_workers "${NUM_WORKERS}" \
  >> "${PIPELINE_LOG}" 2>&1

echo "[PIPELINE] done $(date)" >> "${PIPELINE_LOG}"
