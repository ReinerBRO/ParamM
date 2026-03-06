#!/usr/bin/env bash
set -euo pipefail

PITFALL_PATH="${1:?Usage: run_mbpp_with_lora_pitfalls_orchestrator_miku.sh <pitfall_jsonl> <run_name> [env_name] [num_workers]}"
RUN_NAME="${2:?Usage: run_mbpp_with_lora_pitfalls_orchestrator_miku.sh <pitfall_jsonl> <run_name> [env_name] [num_workers]}"
ENV_NAME="${3:-paramm}"
NUM_WORKERS="${4:-24}"

RUN_ROOT="results/mbpp/paramAgent/${RUN_NAME}"
ORCH_LOG="results/mbpp/paramAgent/${RUN_NAME}_orchestrator.log"

mkdir -p "results/mbpp/paramAgent"
echo "[ORCH] wait_pitfall $(date)" >> "${ORCH_LOG}"
while true; do
  if [ -f "${PITFALL_PATH}" ]; then
    n="$(wc -l < "${PITFALL_PATH}")"
    echo "[ORCH] pitfall_lines=${n} $(date)" >> "${ORCH_LOG}"
    if [ "${n}" -ge 397 ]; then
      break
    fi
  fi
  sleep 20
done

echo "[ORCH] start_phase1 $(date)" >> "${ORCH_LOG}"
PITFALL_PATH="${PITFALL_PATH}" NUM_WORKERS="${NUM_WORKERS}" \
  bash scripts/hpc4/run_mbpp_phase1_sharded_miku.sh "${ENV_NAME}" "${RUN_NAME}" >> "${ORCH_LOG}" 2>&1

echo "[ORCH] start_pipeline $(date)" >> "${ORCH_LOG}"
bash scripts/hpc4/run_mbpp_phase12_pipeline_miku.sh "${RUN_NAME}" "${ENV_NAME}" "${NUM_WORKERS}" >> "${ORCH_LOG}" 2>&1

echo "[ORCH] done $(date)" >> "${ORCH_LOG}"
echo "[ORCH] run_root=${RUN_ROOT}" >> "${ORCH_LOG}"
