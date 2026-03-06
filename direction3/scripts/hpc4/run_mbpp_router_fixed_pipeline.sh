#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/user/user06/ParamAgent/direction3"
RUN_NAME="${1:-dir3_router_mbpp_fix_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-24}"
CONDA_ENV="${CONDA_ENV:-paramm}"
ROUTER_CONF_THRESHOLD="${ROUTER_CONF_THRESHOLD:-0.6}"

RUN_BASE="${PROJECT_ROOT}/results/mbpp_router_runs"
RUN_ROOT="${RUN_BASE}/${RUN_NAME}"
DATASET_PATH="${PROJECT_ROOT}/benchmarks/mbpp-py.jsonl"
PITFALL_PATH="${PITFALL_PATH:-${PROJECT_ROOT}/benchmarks/code_pitfalls/mbpp_pitfalls_lora_20260303_021714.jsonl}"
ROUTER_CKPT="${ROUTER_CKPT:-${PROJECT_ROOT}/results/router_ckpt_miku/router.ckpt}"
PIPELINE_LOG="${RUN_ROOT}/pipeline.log"

mkdir -p "${RUN_ROOT}/shards" "${RUN_ROOT}/workers" "${RUN_ROOT}/logs"
echo "${RUN_ROOT}" > "${RUN_BASE}/LATEST_RUN.txt"

log() {
  printf '%s %s\n' "[PIPELINE]" "$*" | tee -a "${PIPELINE_LOG}"
}

log "run_name=${RUN_NAME}"
log "run_root=${RUN_ROOT}"
log "dataset=${DATASET_PATH}"
log "pitfalls=${PITFALL_PATH}"
log "router_ckpt=${ROUTER_CKPT}"

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

python - <<'PY' "${DATASET_PATH}" "${PITFALL_PATH}" "${RUN_ROOT}/shards" "${NUM_WORKERS}"
import json
import os
import sys

dataset_path, pitfall_path, out_dir, num_workers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

dataset = read_jsonl(dataset_path)
pitfalls = read_jsonl(pitfall_path)
if len(dataset) != len(pitfalls):
    raise SystemExit(f"Dataset and pitfall lengths mismatch: {len(dataset)} vs {len(pitfalls)}")


def norm_prompt(text):
    return " ".join((text or "").split())


def key_ep_prompt(row):
    return (row.get("entry_point"), norm_prompt(row.get("prompt")))


pitfall_map = {}
for i, row in enumerate(pitfalls):
    key = key_ep_prompt(row)
    if key in pitfall_map:
        raise SystemExit(f"Duplicate pitfall key {key!r} at indices {pitfall_map[key]['_idx']} and {i}")
    row["_idx"] = i
    pitfall_map[key] = row

aligned_pitfalls = []
missing = []
for i, row in enumerate(dataset):
    key = key_ep_prompt(row)
    pitfall_row = pitfall_map.pop(key, None)
    if pitfall_row is None:
        missing.append((i, row.get("entry_point")))
        continue
    pitfall_row.pop("_idx", None)
    aligned_pitfalls.append(pitfall_row)

if missing:
    head = ", ".join([f"{idx}:{ep}" for idx, ep in missing[:8]])
    raise SystemExit(f"Missing pitfall match for {len(missing)} dataset rows, sample={head}")
if pitfall_map:
    sample_left = list(pitfall_map.keys())[:5]
    raise SystemExit(f"Unconsumed pitfalls after alignment: {len(pitfall_map)}, sample={sample_left}")

pitfalls = aligned_pitfalls

alignment_summary = {
    "dataset_count": len(dataset),
    "pitfall_count": len(pitfalls),
    "alignment_key": "entry_point+normalized_prompt",
    "aligned": True,
}
with open(os.path.join(out_dir, "alignment_summary.json"), "w", encoding="utf-8") as f:
    json.dump(alignment_summary, f, ensure_ascii=False, indent=2)

n = len(dataset)
for w in range(num_workers):
    start = (w * n) // num_workers
    end = ((w + 1) * n) // num_workers
    write_jsonl(os.path.join(out_dir, f"mbpp_shard_{w:02d}.jsonl"), dataset[start:end])
    write_jsonl(os.path.join(out_dir, f"mbpp_pitfalls_shard_{w:02d}.jsonl"), pitfalls[start:end])
print(f"sharded {n} items across {num_workers} workers")
PY

: > "${RUN_ROOT}/worker_pids.tsv"

for w in $(seq 0 $((NUM_WORKERS - 1))); do
  worker_name="worker_$(printf "%02d" "${w}")"
  ds_shard="${RUN_ROOT}/shards/mbpp_shard_$(printf "%02d" "${w}").jsonl"
  pf_shard="${RUN_ROOT}/shards/mbpp_pitfalls_shard_$(printf "%02d" "${w}").jsonl"
  key_idx=$((2 + w))
  key_var="ZZZ_API_KEY_${key_idx}"
  key_pool="${!key_var}"
  log_file="${RUN_ROOT}/logs/${worker_name}.log"

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
    --phase1_only \
    --verbose > "${log_file}" 2>&1 &

  echo -e "${worker_name}\t$!" >> "${RUN_ROOT}/worker_pids.tsv"
done

log "phase1 launched"

while true; do
  alive=0
  while IFS=$'\t' read -r _ pid; do
    [ -n "${pid:-}" ] || continue
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done < "${RUN_ROOT}/worker_pids.tsv"
  log "phase1_alive=${alive}"
  [ "${alive}" -eq 0 ] && break
  sleep 20
done

PHASE1_MERGE_TIMEOUT="${PHASE1_MERGE_TIMEOUT:-7200}"
PHASE1_SKIP_RECOMPUTE="${PHASE1_SKIP_RECOMPUTE:-1}"
set +e
merge_phase1_extra_args=()
if [ "${PHASE1_SKIP_RECOMPUTE}" = "1" ]; then
  merge_phase1_extra_args+=(--skip_recompute)
fi
PYTHONPATH="${PROJECT_ROOT}" timeout "${PHASE1_MERGE_TIMEOUT}" python "${PROJECT_ROOT}/scripts/hpc4/merge_phase1_shards.py" \
  --run_root "${RUN_ROOT}" \
  --dataset_path "${DATASET_PATH}" \
  --num_workers "${NUM_WORKERS}" \
  "${merge_phase1_extra_args[@]}" >> "${PIPELINE_LOG}" 2>&1
phase1_merge_rc=$?
set -e
if [ "${phase1_merge_rc}" -ne 0 ]; then
  log "phase1 merge rc=${phase1_merge_rc} (timeout=${PHASE1_MERGE_TIMEOUT}s), checking artifacts"
fi
if [ ! -f "${RUN_ROOT}/merged_phase1/mem_bank.pkl" ]; then
  log "phase1 merge failed: missing ${RUN_ROOT}/merged_phase1/mem_bank.pkl"
  exit 1
fi
log "phase1 merge finished (rc=${phase1_merge_rc})"

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

PHASE2_MERGE_TIMEOUT="${PHASE2_MERGE_TIMEOUT:-7200}"
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
if [ ! -f "${RUN_ROOT}/merged_phase2/second_stage_log.jsonl" ]; then
  log "phase2 merge failed: missing ${RUN_ROOT}/merged_phase2/second_stage_log.jsonl"
  exit 1
fi
log "phase2 merge finished (rc=${phase2_merge_rc})"

log "done"
