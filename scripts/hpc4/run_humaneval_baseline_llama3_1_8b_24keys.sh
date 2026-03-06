#!/usr/bin/env bash
set -euo pipefail

# 24-worker HumanEval baseline on miku using llama3_1_8b over relay keys.
#
# Usage:
#   bash scripts/hpc4/run_humaneval_baseline_llama3_1_8b_24keys.sh [conda_env] [run_name] [model_name]
#
# Defaults align with existing results/humaneval naming conventions.

ENV_NAME="${1:-paramm}"
MODEL_NAME="${3:-llama3_1_8b}"
RUN_NAME="${2:-${MODEL_NAME}_humaneval_baseline_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-24}"

PROJECT_ROOT="/data/user/user06/ParamAgent"
DATASET_PATH="benchmarks/humaneval_full.jsonl"
RUN_ROOT="results/humaneval/paramAgent/${RUN_NAME}"
WORKERS_DIR="${RUN_ROOT}/workers"
SHARDS_DIR="${RUN_ROOT}/shards"
LOGS_DIR="${RUN_ROOT}/logs"
PIDS_FILE="${RUN_ROOT}/baseline_worker_pids.tsv"

if [ "${NUM_WORKERS}" -ne 24 ]; then
  echo "[WARN] Recommended NUM_WORKERS=24. Current: ${NUM_WORKERS}"
fi

cd "${PROJECT_ROOT}"

if [ -e "${RUN_ROOT}" ]; then
  echo "[ERROR] Run directory already exists: ${RUN_ROOT}"
  echo "        Please pass a new RUN_NAME."
  exit 1
fi

if [ -f /data/user/user06/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /data/user/user06/miniconda3/etc/profile.d/conda.sh
  conda activate "${ENV_NAME}"
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.zhizengzeng.com/v1}"

# Build strict 24-key pool from ZZZ_API_KEY_2..25.
keys=()
for i in $(seq 2 25); do
  var_name="ZZZ_API_KEY_${i}"
  val="${!var_name:-}"
  if [ -z "${val}" ]; then
    echo "[ERROR] Missing ${var_name} in .env/environment."
    exit 1
  fi
  keys+=("${val}")
done

mkdir -p "${WORKERS_DIR}" "${SHARDS_DIR}" "${LOGS_DIR}"
echo -n > "${PIDS_FILE}"

# Ensure HumanEval visible tests exist (main.py expects this for humaneval).
if [ ! -f benchmarks/humaneval_visible_tests.jsonl ]; then
  python scripts/gen_visible_tests.py
fi

python - <<PY "${DATASET_PATH}" "${SHARDS_DIR}" "${NUM_WORKERS}"
import json
import os
import sys


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
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\\n")


dataset_path = sys.argv[1]
out_dir = sys.argv[2]
num_workers = int(sys.argv[3])
rows = read_jsonl(dataset_path)
n = len(rows)

for w in range(num_workers):
    start = (w * n) // num_workers
    end = ((w + 1) * n) // num_workers
    out = os.path.join(out_dir, f"humaneval_full_shard_{w:02d}.jsonl")
    write_jsonl(out, rows[start:end])
    print(f"worker_{w:02d}: {start}:{end} -> {end-start}")
PY

for w in $(seq 0 $((NUM_WORKERS - 1))); do
  worker_name="worker_$(printf "%02d" "${w}")"
  shard_path="${SHARDS_DIR}/humaneval_full_shard_$(printf "%02d" "${w}").jsonl"
  log_file="${LOGS_DIR}/${worker_name}.log"

  # Build per-worker rotated key pool to avoid all workers starting from key #2.
  rotated=()
  for off in $(seq 0 23); do
    idx=$(( (w + off) % 24 ))
    rotated+=("${keys[$idx]}")
  done
  key_pool="$(IFS=,; echo "${rotated[*]}")"

  OPENAI_API_KEYS="${key_pool}" nohup python main.py \
    --run_name "${worker_name}" \
    --root_dir "${WORKERS_DIR}" \
    --dataset_path "${shard_path}" \
    --strategy simple \
    --language py \
    --model "${MODEL_NAME}" \
    --pass_at_k 1 \
    --max_iters 1 \
    --verbose \
    > "${log_file}" 2>&1 &

  pid=$!
  echo -e "${worker_name}\t${pid}" >> "${PIDS_FILE}"
  echo "Started ${worker_name} pid=${pid} log=${log_file}"
done

echo ""
echo "Submitted baseline run on miku."
echo "Run root: ${RUN_ROOT}"
echo "Workers pid file: ${PIDS_FILE}"
echo "After completion, merge with:"
echo "python scripts/hpc4/merge_humaneval_baseline_shards.py --run_root ${RUN_ROOT} --dataset_path ${DATASET_PATH} --num_workers ${NUM_WORKERS}"
