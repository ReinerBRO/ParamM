#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/user/user06/ParamAgent/direction3"
SOURCE_RUN_NAME="${1:-dir3_router_mbpp_transformer_val_20260306_014139}"
RUN_NAME="${2:-dir3_router_mbpp_transformer_phase2only_hotfix_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-24}"
CONDA_ENV="${CONDA_ENV:-paramm}"
ROUTER_CONF_THRESHOLD="${ROUTER_CONF_THRESHOLD:-0.6}"
ROUTER_CKPT="${ROUTER_CKPT:-${PROJECT_ROOT}/router_checkpoints/transformer_mbpp_v1/router.ckpt}"
DATASET_PATH="${PROJECT_ROOT}/benchmarks/mbpp-py.jsonl"
PHASE2_MERGE_TIMEOUT="${PHASE2_MERGE_TIMEOUT:-7200}"

RUN_BASE="${PROJECT_ROOT}/results/mbpp_router_runs"
SOURCE_RUN_ROOT="${RUN_BASE}/${SOURCE_RUN_NAME}"
RUN_ROOT="${RUN_BASE}/${RUN_NAME}"
PIPELINE_LOG="${RUN_ROOT}/pipeline.log"

log() {
  printf '%s %s\n' "[PHASE2-ONLY]" "$*" | tee -a "${PIPELINE_LOG}"
}

if [ ! -d "${SOURCE_RUN_ROOT}" ]; then
  echo "[ERROR] Source run not found: ${SOURCE_RUN_ROOT}"
  exit 1
fi
if [ ! -d "${SOURCE_RUN_ROOT}/shards" ]; then
  echo "[ERROR] Missing source shards dir: ${SOURCE_RUN_ROOT}/shards"
  exit 1
fi
if [ ! -f "${SOURCE_RUN_ROOT}/merged_phase1/mem_bank.pkl" ]; then
  echo "[ERROR] Missing source merged phase1 mem bank: ${SOURCE_RUN_ROOT}/merged_phase1/mem_bank.pkl"
  exit 1
fi

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/workers"
echo "${RUN_ROOT}" > "${RUN_BASE}/LATEST_RUN.txt"

log "source_run_name=${SOURCE_RUN_NAME}"
log "run_name=${RUN_NAME}"
log "num_workers=${NUM_WORKERS}"
log "router_ckpt=${ROUTER_CKPT}"
log "router_conf_threshold=${ROUTER_CONF_THRESHOLD}"

rm -rf "${RUN_ROOT}/shards" "${RUN_ROOT}/merged_phase1"
ln -s "${SOURCE_RUN_ROOT}/shards" "${RUN_ROOT}/shards"
ln -s "${SOURCE_RUN_ROOT}/merged_phase1" "${RUN_ROOT}/merged_phase1"

for w in $(seq 0 $((NUM_WORKERS - 1))); do
  worker_name="worker_$(printf "%02d" "${w}")"
  src_worker="${SOURCE_RUN_ROOT}/workers/${worker_name}"
  dst_worker="${RUN_ROOT}/workers/${worker_name}"
  mkdir -p "${dst_worker}"

  if [ ! -f "${src_worker}/first_stage_log.jsonl" ]; then
    echo "[ERROR] Missing source first_stage_log: ${src_worker}/first_stage_log.jsonl"
    exit 1
  fi
  rm -f "${dst_worker}/first_stage_log.jsonl" "${dst_worker}/failed_probs.pkl"
  ln -s "${src_worker}/first_stage_log.jsonl" "${dst_worker}/first_stage_log.jsonl"
  if [ -f "${src_worker}/failed_probs.pkl" ]; then
    ln -s "${src_worker}/failed_probs.pkl" "${dst_worker}/failed_probs.pkl"
  fi
done

source /data/user/user06/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

if [ -f "${PROJECT_ROOT}/.env" ]; then
  set -a
  source "${PROJECT_ROOT}/.env"
  set +a
fi

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.zhizengzeng.com/v1}"
export RELAY_FALLBACK_MODEL=""
log "relay_fallback_model=DISABLED"

for i in $(seq 2 25); do
  var_name="ZZZ_API_KEY_${i}"
  if [ -z "${!var_name:-}" ]; then
    log "missing ${var_name}"
    exit 1
  fi
done

: > "${RUN_ROOT}/phase2_worker_pids.tsv"
GLOBAL_MEM_BANK="${RUN_ROOT}/merged_phase1/mem_bank.pkl"

for w in $(seq 0 $((NUM_WORKERS - 1))); do
  worker_name="worker_$(printf "%02d" "${w}")"
  ds_shard="${RUN_ROOT}/shards/mbpp_shard_$(printf "%02d" "${w}").jsonl"
  pf_shard="${RUN_ROOT}/shards/mbpp_pitfalls_shard_$(printf "%02d" "${w}").jsonl"
  key_idx=$((2 + w))
  key_var="ZZZ_API_KEY_${key_idx}"
  key_pool="${!key_var}"
  log_file="${RUN_ROOT}/logs/${worker_name}_phase2.log"

  if [ ! -f "${ds_shard}" ] || [ ! -f "${pf_shard}" ]; then
    echo "[ERROR] Missing shard inputs for ${worker_name}"
    exit 1
  fi

  OPENAI_API_KEYS="${key_pool}" nohup python "${PROJECT_ROOT}/main_param.py" \
    --run_name "${worker_name}" \
    --root_dir "${RUN_ROOT}/workers" \
    --dataset_path "${ds_shard}" \
    --strategy dot \
    --language py \
    --model llama3_1_8b \
    --pass_at_k 1 \
    --max_iters 5 \
    --inner_iter 5 \
    --use_mistakes \
    --mistake_json_path "${pf_shard}" \
    --fix_stage1_indices \
    --global_mem_bank_path "${GLOBAL_MEM_BANK}" \
    --router_enable \
    --router_conf_threshold "${ROUTER_CONF_THRESHOLD}" \
    --router_ckpt_path "${ROUTER_CKPT}" \
    --verbose > "${log_file}" 2>&1 &

  echo -e "${worker_name}\t$!" >> "${RUN_ROOT}/phase2_worker_pids.tsv"
done

log "phase2 launched (router ON)"

while true; do
  alive=0
  while IFS=$'\t' read -r _ pid; do
    [ -n "${pid:-}" ] || continue
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done < "${RUN_ROOT}/phase2_worker_pids.tsv"
  log "phase2_alive=${alive}"
  [ "${alive}" -eq 0 ] && break
  sleep 20
done

set +e
PYTHONPATH="${PROJECT_ROOT}" timeout "${PHASE2_MERGE_TIMEOUT}" python "${PROJECT_ROOT}/scripts/hpc4/merge_phase2_shards.py" \
  --run_root "${RUN_ROOT}" \
  --dataset_path "${DATASET_PATH}" \
  --num_workers "${NUM_WORKERS}" >> "${PIPELINE_LOG}" 2>&1
phase2_merge_rc=$?
set -e
if [ "${phase2_merge_rc}" -ne 0 ]; then
  log "phase2 merge rc=${phase2_merge_rc} (timeout=${PHASE2_MERGE_TIMEOUT}s), checking artifacts"
fi

if [ ! -f "${RUN_ROOT}/results.txt" ]; then
  log "missing results.txt after merge"
  exit 1
fi

log "completed"
cat "${RUN_ROOT}/results.txt"
