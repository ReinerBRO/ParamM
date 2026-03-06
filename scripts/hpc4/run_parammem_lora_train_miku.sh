#!/usr/bin/env bash
set -euo pipefail

# Table-1 style LoRA training launcher on miku (Ascend NPU, 8 chips)
# Usage:
#   bash scripts/hpc4/run_parammem_lora_train_miku.sh
#   NPU_CHIPS=0,1,2,3,4,5,6,7 bash scripts/hpc4/run_parammem_lora_train_miku.sh

MODEL_PATH="${MODEL_PATH:-/data/user/user06/cache/Models/Llama-3.1-8B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-/data/user/user06/data/paramagent/table1_lora_train/parammem_train_merged_clean.jsonl}"
RUN_TAG="${RUN_TAG:-table1_llama31_8b_parammem_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/data/user/user06/data/paramagent/lora_runs}"
OUT_DIR="${OUT_ROOT}/${RUN_TAG}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/opt/conda/public/envs/ms-swift312-npu}"
LOG_ROOT="${LOG_ROOT:-/data/user/user06/ParamAgent/logs}"

NPU_CHIPS="${NPU_CHIPS:-0,1,2,3,4,5,6,7}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29600}"

# Paper-aligned core params
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LR="${LR:-2e-5}"
EPOCHS="${EPOCHS:-3}"

PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SEED="${SEED:-42}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"

mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_ROOT}"

export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_PATH}" || true

IFS=, read -r -a NPU_ARR <<< "${NPU_CHIPS}"
NPROC_PER_NODE="${#NPU_ARR[@]}"

echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] TRAIN_JSONL=${TRAIN_JSONL}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] NPU_CHIPS=${NPU_CHIPS} (nproc=${NPROC_PER_NODE})"
echo "[INFO] python=$(which python)"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_CHIPS}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRAIN_JSONL

TS="$(date +%Y%m%d_%H%M%S)"
MODEL_NAME="$(basename "${MODEL_PATH}" | tr ' /' '__')"
DATASET_NAME="$(basename "${TRAIN_JSONL}" .jsonl | tr ' /' '__')"
LOG_FILE="${LOG_ROOT}/${MODEL_NAME}_${DATASET_NAME}_${TS}.log"
PID_FILE="${OUT_DIR}/train.pid"
EXTRA_ARGS=()
if [ "${MAX_TRAIN_SAMPLES}" -gt 0 ]; then
  EXTRA_ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")
fi

python - <<PY
import os
p=os.environ.get("TRAIN_JSONL")
if not p or not os.path.exists(p):
    raise SystemExit(f"[ERROR] TRAIN_JSONL not found: {p}")
print(f"[CHECK] train file exists: {p}")
PY

nohup torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  scripts/train/train_parammem_lora.py \
  --model_path "${MODEL_PATH}" \
  --train_jsonl "${TRAIN_JSONL}" \
  --output_dir "${OUT_DIR}" \
  --seed "${SEED}" \
  --max_length "${MAX_LENGTH}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --per_device_batch_size "${PER_DEVICE_BATCH}" \
  --grad_accum "${GRAD_ACCUM}" \
  --save_steps 100 \
  --log_steps 1 \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout 0.0 \
  --device_backend npu \
  --bf16 \
  "${EXTRA_ARGS[@]}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

echo "[OK] started. pid=${PID}"
echo "[OK] log: ${LOG_FILE}"
echo "[OK] pid: ${PID_FILE}"
