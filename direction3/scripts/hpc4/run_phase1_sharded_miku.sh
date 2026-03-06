#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-paramm}"
RUN_NAME="${2:-llama3_1_8b_humaneval_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-24}"

PROJECT_ROOT="/data/user/user06/ParamAgent"
DATASET_PATH="benchmarks/humaneval_full.jsonl"
PITFALL_PATH="${PITFALL_PATH:-benchmarks/code_pitfalls/humaneval_full_pitfalls.jsonl}"
ROOT_DIR="results/humaneval/paramAgent"
RUN_ROOT="${ROOT_DIR}/${RUN_NAME}"

if [ "${NUM_WORKERS}" -ne 24 ]; then
  echo "[WARN] This script is configured for 24 workers; got NUM_WORKERS=${NUM_WORKERS}."
fi

cd "${PROJECT_ROOT}"

if [ -e "${RUN_ROOT}" ]; then
  echo "[ERROR] Run directory already exists: ${RUN_ROOT}"
  echo "        Please pass a new RUN_NAME to avoid overwriting old runs."
  exit 1
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
source /data/user/user06/miniconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

set -a
source ./.env
set +a

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.zhizengzeng.com/v1}"

for i in $(seq 2 25); do
  var_name="ZZZ_API_KEY_${i}"
  if [ -z "${!var_name:-}" ]; then
    echo "[ERROR] Missing ${var_name} in environment/.env"
    exit 1
  fi
done

export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64
export UV_THREADPOOL_SIZE=64

mkdir -p "${RUN_ROOT}/shards" "${RUN_ROOT}/workers" "${RUN_ROOT}/logs"

python - <<PY "${DATASET_PATH}" "${PITFALL_PATH}" "${RUN_ROOT}/shards" "${NUM_WORKERS}"
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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")


dataset_path = sys.argv[1]
pitfall_path = sys.argv[2]
out_dir = sys.argv[3]
num_workers = int(sys.argv[4])

dataset = read_jsonl(dataset_path)
pitfalls = read_jsonl(pitfall_path)
if len(dataset) != len(pitfalls):
    raise SystemExit(f"Dataset and pitfall lengths mismatch: {len(dataset)} vs {len(pitfalls)}")

n = len(dataset)
for w in range(num_workers):
    start = (w * n) // num_workers
    end = ((w + 1) * n) // num_workers
    ds_out = os.path.join(out_dir, f"humaneval_full_shard_{w:02d}.jsonl")
    pf_out = os.path.join(out_dir, f"humaneval_full_pitfalls_shard_{w:02d}.jsonl")
    write_jsonl(ds_out, dataset[start:end])
    write_jsonl(pf_out, pitfalls[start:end])
    print(f"worker_{w:02d}: {start}:{end} -> {end-start} items")
PY

python scripts/gen_visible_tests.py

PIDS_FILE="${RUN_ROOT}/worker_pids.tsv"
echo -n > "${PIDS_FILE}"

for w in $(seq 0 $((NUM_WORKERS - 1))); do
  worker_name="worker_$(printf "%02d" "${w}")"
  ds_shard="${RUN_ROOT}/shards/humaneval_full_shard_$(printf "%02d" "${w}").jsonl"
  pf_shard="${RUN_ROOT}/shards/humaneval_full_pitfalls_shard_$(printf "%02d" "${w}").jsonl"

  key_idx=$((2 + w))
  key_var="ZZZ_API_KEY_${key_idx}"
  key_pool="${!key_var}"

  log_file="${RUN_ROOT}/logs/${worker_name}.log"

  OPENAI_API_KEYS="${key_pool}" nohup python main_param.py \
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
    --verbose \
    > "${log_file}" 2>&1 &

  pid=$!
  echo -e "${worker_name}\t${pid}" >> "${PIDS_FILE}"
  echo "Started ${worker_name} pid=${pid} key=${key_var} log=${log_file}"
done

echo ""
echo "Phase-1 sharded run started."
echo "Run root: ${RUN_ROOT}"
echo "PID file: ${PIDS_FILE}"
echo "After workers finish, run merge:"
echo "python scripts/hpc4/merge_phase1_shards.py --run_root ${RUN_ROOT} --dataset_path ${DATASET_PATH} --num_workers ${NUM_WORKERS}"
