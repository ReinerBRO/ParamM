#!/usr/bin/env bash
set -euo pipefail

SRC_RUN="${1:?Usage: mbpp_wait_clean_rerun_miku.sh <src_run_name> <src_pitfall_jsonl> <rerun_name> [env_name] [num_workers]}"
SRC_PIT="${2:?Usage: mbpp_wait_clean_rerun_miku.sh <src_run_name> <src_pitfall_jsonl> <rerun_name> [env_name] [num_workers]}"
RERUN_NAME="${3:?Usage: mbpp_wait_clean_rerun_miku.sh <src_run_name> <src_pitfall_jsonl> <rerun_name> [env_name] [num_workers]}"
ENV_NAME="${4:-paramm}"
NUM_WORKERS="${5:-24}"
PROJECT_ROOT="/data/user/user06/ParamAgent"

SRC_ROOT="results/mbpp/paramAgent/${SRC_RUN}"
RERUN_ROOT="results/mbpp/paramAgent/${RERUN_NAME}"
LOG_PATH="results/mbpp/paramAgent/${RERUN_NAME}_auto.log"
CLEAN_PIT="${SRC_PIT%.jsonl}_clean.jsonl"
REPORT_JSON="${SRC_PIT%.jsonl}_clean_report.json"

mkdir -p "results/mbpp/paramAgent"
source /data/user/user06/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "[AUTO] wait_src_done $(date)" >> "${LOG_PATH}"
while true; do
  if [ -f "${SRC_ROOT}/results.txt" ] && [ -f "${SRC_ROOT}/merged_phase2/merge_summary.json" ]; then
    break
  fi
  p1_alive=0
  p2_alive=0
  if [ -f "${SRC_ROOT}/worker_pids.tsv" ]; then
    while IFS=$'\t' read -r _ pid; do
      [ -n "${pid:-}" ] || continue
      if kill -0 "${pid}" 2>/dev/null; then
        p1_alive=$((p1_alive + 1))
      fi
    done < "${SRC_ROOT}/worker_pids.tsv"
  fi
  if [ -f "${SRC_ROOT}/phase2_worker_pids.tsv" ]; then
    while IFS=$'\t' read -r _ pid; do
      [ -n "${pid:-}" ] || continue
      if kill -0 "${pid}" 2>/dev/null; then
        p2_alive=$((p2_alive + 1))
      fi
    done < "${SRC_ROOT}/phase2_worker_pids.tsv"
  fi
  echo "[AUTO] src_alive phase1=${p1_alive} phase2=${p2_alive} $(date)" >> "${LOG_PATH}"
  sleep 30
done

echo "[AUTO] src_done $(date)" >> "${LOG_PATH}"
"${PYTHON_BIN}" scripts/data/clean_pitfalls_jsonl.py \
  --input_jsonl "${SRC_PIT}" \
  --output_jsonl "${CLEAN_PIT}" \
  --report_json "${REPORT_JSON}" \
  >> "${LOG_PATH}" 2>&1

echo "[AUTO] clean_done clean_pit=${CLEAN_PIT} $(date)" >> "${LOG_PATH}"
PITFALL_PATH="${CLEAN_PIT}" NUM_WORKERS="${NUM_WORKERS}" \
  bash scripts/hpc4/run_mbpp_phase1_sharded_miku.sh "${ENV_NAME}" "${RERUN_NAME}" >> "${LOG_PATH}" 2>&1

echo "[AUTO] rerun_phase1_started $(date)" >> "${LOG_PATH}"
bash scripts/hpc4/run_mbpp_phase12_pipeline_miku.sh "${RERUN_NAME}" "${ENV_NAME}" "${NUM_WORKERS}" >> "${LOG_PATH}" 2>&1
echo "[AUTO] rerun_done $(date)" >> "${LOG_PATH}"
echo "[AUTO] rerun_root=${RERUN_ROOT}" >> "${LOG_PATH}"
