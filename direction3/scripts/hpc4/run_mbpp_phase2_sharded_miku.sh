#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-paramm}"
RUN_NAME="${2:-llama3_1_8b_mbpp_20260303_000000}"
NUM_WORKERS="${NUM_WORKERS:-24}"

PROJECT_ROOT="/data/user/user06/ParamAgent"
RUN_ROOT="results/mbpp/paramAgent/${RUN_NAME}"
SHARDS_DIR="${RUN_ROOT}/shards"
WORKERS_DIR="${RUN_ROOT}/workers"
LOGS_DIR="${RUN_ROOT}/logs"
GLOBAL_MEM_BANK="${RUN_ROOT}/merged_phase1/mem_bank.pkl"
DATASET_PATH="benchmarks/mbpp-py.jsonl"

if [ "${NUM_WORKERS}" -ne 24 ]; then
  echo "[WARN] This script is configured for 24 workers; got NUM_WORKERS=${NUM_WORKERS}."
fi

cd "${PROJECT_ROOT}"

if [ ! -d "${RUN_ROOT}" ]; then
  echo "[ERROR] Run root does not exist: ${RUN_ROOT}"
  exit 1
fi
if [ ! -d "${SHARDS_DIR}" ]; then
  echo "[ERROR] Shards directory missing: ${SHARDS_DIR}"
  exit 1
fi
if [ ! -f "${GLOBAL_MEM_BANK}" ]; then
  echo "[ERROR] Global phase1 merged mem bank not found: ${GLOBAL_MEM_BANK}"
  echo "        Please run phase1 merge first."
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

mkdir -p "${WORKERS_DIR}" "${LOGS_DIR}"

PIDS_FILE="${RUN_ROOT}/phase2_worker_pids.tsv"
echo -n > "${PIDS_FILE}"

for w in $(seq 0 $((NUM_WORKERS - 1))); do
  worker_name="worker_$(printf "%02d" "${w}")"
  worker_dir="${WORKERS_DIR}/${worker_name}"
  ds_shard="${SHARDS_DIR}/mbpp_shard_$(printf "%02d" "${w}").jsonl"
  pf_shard="${SHARDS_DIR}/mbpp_pitfalls_shard_$(printf "%02d" "${w}").jsonl"

  if [ ! -d "${worker_dir}" ]; then
    echo "[ERROR] Missing worker dir: ${worker_dir}"
    exit 1
  fi
  if [ ! -f "${worker_dir}/first_stage_log.jsonl" ]; then
    echo "[ERROR] Missing first_stage_log for ${worker_name}: ${worker_dir}/first_stage_log.jsonl"
    exit 1
  fi
  if [ ! -f "${ds_shard}" ] || [ ! -f "${pf_shard}" ]; then
    echo "[ERROR] Missing shard inputs for ${worker_name}"
    exit 1
  fi

  key_idx=$((2 + w))
  key_var="ZZZ_API_KEY_${key_idx}"
  key_pool="${!key_var}"
  log_file="${LOGS_DIR}/${worker_name}_phase2.log"

  OPENAI_API_KEYS="${key_pool}" nohup python main_param.py \
    --run_name "${worker_name}" \
    --root_dir "${WORKERS_DIR}" \
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
    --verbose \
    > "${log_file}" 2>&1 &

  pid=$!
  echo -e "${worker_name}\t${pid}" >> "${PIDS_FILE}"
  echo "Started ${worker_name} pid=${pid} key=${key_var} log=${log_file}"
done

echo ""
echo "MBPP phase-2 sharded run started."
echo "Run root: ${RUN_ROOT}"
echo "PID file: ${PIDS_FILE}"
echo "After workers finish, run merge:"
echo "python scripts/hpc4/merge_phase2_shards.py --run_root ${RUN_ROOT} --dataset_path ${DATASET_PATH} --num_workers ${NUM_WORKERS}"
