#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-paramm}"
RUN_NAME="${2:-gpt4omini_parammem_synth_$(date +%Y%m%d_%H%M%S)}"
APPS_SOURCE_JSONL="${APPS_SOURCE_JSONL:-}"

PROJECT_ROOT="/data/user/user06/ParamAgent"
DEV_NODE="miku"
OUT_ROOT="results/parammem_data/${RUN_NAME}"
OUT_JSONL="${OUT_ROOT}/parammem_code_supervision_8200.jsonl"

ssh "$DEV_NODE" \
  ENV_NAME="$ENV_NAME" \
  PROJECT_ROOT="$PROJECT_ROOT" \
  OUT_ROOT="$OUT_ROOT" \
  OUT_JSONL="$OUT_JSONL" \
  APPS_SOURCE_JSONL="$APPS_SOURCE_JSONL" \
  DEV_NODE="$DEV_NODE" \
  bash -s <<'REMOTE'
set -euo pipefail
cd "$PROJECT_ROOT"

set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
set -u
source /data/user/user06/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
set -a
source ./.env
set +a

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.zhizengzeng.com/v1}"

for i in $(seq 2 25); do
  var_name="ZZZ_API_KEY_${i}"
  if [ -z "${!var_name:-}" ]; then
    echo "[ERROR] Missing ${var_name}"
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

CMD=(python scripts/data/synthesize_parammem_code_dataset.py
  --apps_samples 4000
  --synth_samples 4200
  --workers 24
  --model gpt-4o-mini
  --temperature 0.7
  --max_tokens 1024
  --max_retries 4
  --seed 42
  --output_jsonl "$OUT_JSONL"
  --work_dir "$OUT_ROOT/work")

if [ -n "$APPS_SOURCE_JSONL" ]; then
  CMD+=(--apps_source_jsonl "$APPS_SOURCE_JSONL")
fi

nohup "${CMD[@]}" > "$OUT_ROOT/synthesis_nohup.log" 2>&1 &
echo $! > "$OUT_ROOT/synthesis.pid"

echo "Started synthesis on $DEV_NODE"
echo "Run dir: $OUT_ROOT"
echo "PID: $(cat "$OUT_ROOT/synthesis.pid")"
echo "Log: $OUT_ROOT/synthesis_nohup.log"
REMOTE
