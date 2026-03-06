#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-/data/user/user06/cache/Models/Llama-3.1-8B-Instruct}"
LORA_PATH="${LORA_PATH:-/data/user/user06/data/paramagent/lora_runs/table1_llama31_8b_parammem_20260302_200901}"
INPUT_JSONL="${INPUT_JSONL:-/data/user/user06/ParamAgent/benchmarks/humaneval_full.jsonl}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-/data/user/user06/data/paramagent/pitfall_runs/humaneval_lora_${RUN_TS}}"
OUT_JSONL="${OUT_JSONL:-/data/user/user06/ParamAgent/benchmarks/code_pitfalls/humaneval_full_pitfalls_lora_${RUN_TS}.jsonl}"
LOG_DIR="${LOG_DIR:-/data/user/user06/ParamAgent/logs}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/opt/conda/public/envs/ms-swift312-npu}"
CHIPS="${CHIPS:-0,1,2,3,4,5,6,7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-600}"

mkdir -p "$RUN_DIR" "$LOG_DIR"
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /opt/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_PATH" || true

IFS=, read -r -a CHIP_ARR <<< "$CHIPS"
N=${#CHIP_ARR[@]}

echo "[INFO] chips=$CHIPS shard_total=$N"
echo "[INFO] run_dir=$RUN_DIR"
echo "[INFO] out_jsonl=$OUT_JSONL"

pids=()
for idx in "${!CHIP_ARR[@]}"; do
  chip="${CHIP_ARR[$idx]}"
  shard_out="$RUN_DIR/shard_${idx}.jsonl"
  shard_log="$LOG_DIR/pitfall_shard${idx}_${RUN_TS}.log"
  (
    export ASCEND_RT_VISIBLE_DEVICES="$chip"
    export HCCL_CONNECT_TIMEOUT=1800
    export TOKENIZERS_PARALLELISM=false
    export PYTHONUNBUFFERED=1
    python /data/user/user06/ParamAgent/scripts/data/generate_humaneval_pitfalls_lora_npu.py \
      --base_model "$BASE_MODEL" \
      --lora_path "$LORA_PATH" \
      --input_jsonl "$INPUT_JSONL" \
      --output_jsonl "$shard_out" \
      --shard_idx "$idx" \
      --shard_total "$N" \
      --max_new_tokens "$MAX_NEW_TOKENS"
  ) >"$shard_log" 2>&1 &
  pids+=("$!")
  echo "[START] shard=$idx chip=$chip pid=${pids[-1]} log=$shard_log"
done

fail=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "[ERROR] one or more shards failed"
  exit 1
fi

python - <<PY
import json
from pathlib import Path

def row_uid(r):
    for k in ("task_id", "name", "question_id", "entry_point", "prompt"):
        v = r.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""

run_dir=Path('$RUN_DIR')
out_path=Path('$OUT_JSONL')
rows={}
for p in sorted(run_dir.glob('shard_*.jsonl')):
    if not p.exists():
        continue
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r=json.loads(line)
        uid=row_uid(r)
        if uid:
            rows[uid]=r
merged=[rows[k] for k in sorted(rows.keys())]
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open('w',encoding='utf-8') as f:
    for r in merged:
        f.write(json.dumps(r,ensure_ascii=False)+'\\n')
print({'merged':len(merged),'output':str(out_path)})
PY

echo "[OK] merged -> $OUT_JSONL"
