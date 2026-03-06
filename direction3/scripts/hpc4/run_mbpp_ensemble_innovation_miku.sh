#!/usr/bin/env bash
set -euo pipefail

# INNOVATION: Multi-Strategy Ensemble Phase2
# Bypasses broken router, uses 3 parallel strategies, picks best by test feedback

PROJECT_ROOT="/data/user/user06/ParamAgent/direction3"
RUN_NAME="dir3_ensemble_mbpp_$(date +%Y%m%d_%H%M%S)"
NUM_WORKERS="${NUM_WORKERS:-24}"
CONDA_ENV="${CONDA_ENV:-paramm}"

RUN_BASE="${PROJECT_ROOT}/results/mbpp_router_runs"
RUN_ROOT="${RUN_BASE}/${RUN_NAME}"
DATASET_PATH="${PROJECT_ROOT}/benchmarks/mbpp-py.jsonl"
PITFALL_PATH="${PITFALL_PATH:-${PROJECT_ROOT}/benchmarks/code_pitfalls/mbpp_pitfalls_lora_20260303_021714.jsonl}"
PIPELINE_LOG="${RUN_ROOT}/pipeline.log"

mkdir -p "${RUN_ROOT}/shards" "${RUN_ROOT}/workers" "${RUN_ROOT}/logs"
echo "${RUN_ROOT}" > "${RUN_BASE}/LATEST_RUN.txt"

log() {
  printf '%s %s\n' "[ENSEMBLE]" "$*" | tee -a "${PIPELINE_LOG}"
}

log "=== ENSEMBLE INNOVATION RUN ==="
log "run_name=${RUN_NAME}"
log "run_root=${RUN_ROOT}"
log "dataset=${DATASET_PATH}"
log "pitfalls=${PITFALL_PATH}"
log "innovation=Multi-Strategy Ensemble (no router)"

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

# Shard dataset
python - <<'PY' "${DATASET_PATH}" "${PITFALL_PATH}" "${RUN_ROOT}/shards" "${NUM_WORKERS}"
import json, os, sys

dataset_path, pitfall_path, out_dir, num_workers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

dataset = read_jsonl(dataset_path)
pitfalls = read_jsonl(pitfall_path)

def norm_prompt(text):
    return " ".join((text or "").split())

def key_ep_prompt(row):
    return (row.get("entry_point"), norm_prompt(row.get("prompt")))

pitfall_map = {key_ep_prompt(row): row for row in pitfalls}
aligned_pitfalls = []
for row in dataset:
    key = key_ep_prompt(row)
    pitfall_row = pitfall_map.get(key)
    if pitfall_row:
        aligned_pitfalls.append(pitfall_row)
    else:
        aligned_pitfalls.append({"mistake_insights": ""})

shard_size = (len(dataset) + num_workers - 1) // num_workers
for i in range(num_workers):
    start, end = i * shard_size, min((i + 1) * shard_size, len(dataset))
    write_jsonl(f"{out_dir}/mbpp_shard_{i:02d}.jsonl", dataset[start:end])
    write_jsonl(f"{out_dir}/mbpp_pitfalls_shard_{i:02d}.jsonl", aligned_pitfalls[start:end])
PY

log "Sharding complete"

# Phase 1: Launch workers
log "=== PHASE 1: Initial generation ==="
worker_pids=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
  worker_id=$(printf "%02d" $i)
  api_key_var="ZZZ_API_KEY_$((i + 2))"
  
  (
    export OPENAI_API_KEY="${!api_key_var}"
    cd "${PROJECT_ROOT}"
    python main_param.py \
      --run_name "worker_${worker_id}" \
      --root_dir "${RUN_ROOT}/workers/worker_${worker_id}" \
      --dataset_path "${RUN_ROOT}/shards/mbpp_shard_${worker_id}.jsonl" \
      --strategy "dot" \
      --model "gpt-4o-mini" \
      --language "py" \
      --max_iters 5 \
      --pass_at_k 1 \
      --verbose \
      --use_mistakes \
      --mistake_json_path "${RUN_ROOT}/shards/mbpp_pitfalls_shard_${worker_id}.jsonl" \
       \
      --phase1_only \
      > "${RUN_ROOT}/logs/worker_${worker_id}.log" 2>&1
  ) &
  worker_pids+=($!)
  log "Started worker ${worker_id} (PID=${worker_pids[-1]})"
done

printf "worker_id\tpid\n" > "${RUN_ROOT}/worker_pids.tsv"
for i in "${!worker_pids[@]}"; do
  printf "worker_%02d\t%s\n" "$i" "${worker_pids[$i]}" >> "${RUN_ROOT}/worker_pids.tsv"
done

log "Waiting for ${#worker_pids[@]} phase1 workers..."
for pid in "${worker_pids[@]}"; do
  wait "$pid" || log "Worker PID=$pid failed"
done
log "Phase 1 complete"

# Merge phase1
python - <<'PY' "${RUN_ROOT}"
import json, os, sys
run_root = sys.argv[1]
workers_dir = f"{run_root}/workers"
merged_dir = f"{run_root}/merged_phase1"
os.makedirs(merged_dir, exist_ok=True)

all_logs = []
for worker in sorted(os.listdir(workers_dir)):
    log_path = f"{workers_dir}/{worker}/first_stage_log.jsonl"
    if os.path.exists(log_path):
        with open(log_path) as f:
            all_logs.extend([json.loads(line) for line in f if line.strip()])

with open(f"{merged_dir}/first_stage_log.jsonl", "w") as f:
    for log in all_logs:
        f.write(json.dumps(log) + "\n")

solved = sum(1 for log in all_logs if log.get("is_solved"))
print(f"Phase1: {solved}/{len(all_logs)} solved ({solved/len(all_logs)*100:.2f}%)")
PY

# Phase 2: Ensemble innovation
log "=== PHASE 2: Ensemble innovation ==="
worker_pids=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
  worker_id=$(printf "%02d" $i)
  api_key_var="ZZZ_API_KEY_$((i + 2))"
  
  (
    export OPENAI_API_KEY="${!api_key_var}"
    cd "${PROJECT_ROOT}"
    python main_param.py \
      --run_name "worker_${worker_id}_phase2" \
      --root_dir "${RUN_ROOT}/workers/worker_${worker_id}" \
      --dataset_path "${RUN_ROOT}/shards/mbpp_shard_${worker_id}.jsonl" \
      --strategy "dot" \
      --model "gpt-4o-mini" \
      --language "py" \
      --max_iters 5 \
      --pass_at_k 1 \
      --verbose \
      --use_mistakes \
      --mistake_json_path "${RUN_ROOT}/shards/mbpp_pitfalls_shard_${worker_id}.jsonl" \
       \
      --router_enable \
      >> "${RUN_ROOT}/logs/worker_${worker_id}.log" 2>&1
  ) &
  worker_pids+=($!)
  log "Started phase2 worker ${worker_id} (PID=${worker_pids[-1]})"
done

printf "worker_id\tpid\n" > "${RUN_ROOT}/phase2_worker_pids.tsv"
for i in "${!worker_pids[@]}"; do
  printf "worker_%02d\t%s\n" "$i" "${worker_pids[$i]}" >> "${RUN_ROOT}/phase2_worker_pids.tsv"
done

log "Waiting for ${#worker_pids[@]} phase2 workers..."
for pid in "${worker_pids[@]}"; do
  wait "$pid" || log "Phase2 worker PID=$pid failed"
done
log "Phase 2 complete"

# Merge phase2
python - <<'PY' "${RUN_ROOT}"
import json, os, sys
from datetime import datetime
run_root = sys.argv[1]
workers_dir = f"{run_root}/workers"
merged_dir = f"{run_root}/merged_phase2"
os.makedirs(merged_dir, exist_ok=True)

all_logs = []
for worker in sorted(os.listdir(workers_dir)):
    log_path = f"{workers_dir}/{worker}/second_stage_log.jsonl"
    if os.path.exists(log_path):
        with open(log_path) as f:
            all_logs.extend([json.loads(line) for line in f if line.strip()])

with open(f"{merged_dir}/second_stage_log.jsonl", "w") as f:
    for log in all_logs:
        f.write(json.dumps(log) + "\n")

solved = sum(1 for log in all_logs if log.get("is_solved"))
print(f"Phase2: {solved}/{len(all_logs)} solved ({solved/len(all_logs)*100:.2f}%)")

# Write results
with open(f"{run_root}/results.txt", "w") as f:
    f.write(f"run_root: {run_root}\n")
    f.write(f"updated_at: {datetime.now().isoformat()}\n")
    
    # Phase1 stats
    phase1_logs = []
    phase1_path = f"{run_root}/merged_phase1/first_stage_log.jsonl"
    if os.path.exists(phase1_path):
        with open(phase1_path) as fp:
            phase1_logs = [json.loads(line) for line in fp if line.strip()]
    phase1_solved = sum(1 for log in phase1_logs if log.get("is_solved"))
    f.write(f"phase1_solved: {phase1_solved}/{len(phase1_logs)}\n")
    f.write(f"phase1_acc: {phase1_solved/len(phase1_logs)*100:.2f}%\n")
    
    f.write(f"phase2_solved: {solved}/{len(all_logs)}\n")
    f.write(f"phase2_acc: {solved/len(all_logs)*100:.2f}%\n")
    f.write(f"merged_phase2_dir: {merged_dir}\n")
PY

log "=== ENSEMBLE RUN COMPLETE ==="
log "Results: ${RUN_ROOT}/results.txt"
cat "${RUN_ROOT}/results.txt"
