#!/usr/bin/env bash
set -euo pipefail

PROJECT="/data/user/user06/ParamAgent/direction1"
ROOT_PROJECT="/data/user/user06/ParamAgent"
RUN_TAG="direction1_qwen2_1p5b_humaneval_$(date +%Y%m%d_%H%M%S)"
NUM_WORKERS=24

HE_RUN_ROOT="$PROJECT/results/humaneval/paramAgent/${RUN_TAG}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh || true
source /data/user/user06/miniconda3/etc/profile.d/conda.sh
conda activate paramm

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

set -a
source "$ROOT_PROJECT/.env"
set +a

for i in $(seq 2 49); do
  var="ZZZ_API_KEY_${i}"
  if [ -z "${!var:-}" ]; then
    echo "[WARN] missing $var"
  fi
done

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.zhizengzeng.com/v1}"
export PYTHONPATH="$ROOT_PROJECT:$PROJECT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=64
export MKL_NUM_THREADS=64
export OPENBLAS_NUM_THREADS=64
export NUMEXPR_NUM_THREADS=64
export UV_THREADPOOL_SIZE=64

mkdir -p "$HE_RUN_ROOT/shards" "$HE_RUN_ROOT/workers" "$HE_RUN_ROOT/logs"

cd "$PROJECT"

# Generate shards
python - <<PY "$ROOT_PROJECT/benchmarks/humaneval_full.jsonl" "$ROOT_PROJECT/benchmarks/code_pitfalls/humaneval_full_pitfalls.jsonl" "$HE_RUN_ROOT/shards" "$NUM_WORKERS"
import json, os, sys

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

echo "[ORCH] Starting Phase1 for ${RUN_TAG}"
PIDS_FILE="$HE_RUN_ROOT/logs/phase1_pids.tsv"
: > "$PIDS_FILE"

for w in $(seq 0 $((NUM_WORKERS - 1))); do
  worker="worker_$(printf "%02d" "$w")"
  ds_shard="$HE_RUN_ROOT/shards/humaneval_full_shard_$(printf "%02d" "$w").jsonl"
  pf_shard="$HE_RUN_ROOT/shards/humaneval_full_pitfalls_shard_$(printf "%02d" "$w").jsonl"
  log_file="$HE_RUN_ROOT/logs/${worker}_phase1.log"

  # Each worker uses 2 API keys (48 keys / 24 workers = 2 keys per worker)
  key_start=$((2 + w * 2))
  key_pool=""
  for k in $(seq 0 1); do
    key_idx=$((key_start + k))
    key_var="ZZZ_API_KEY_${key_idx}"
    if [ -n "${key_pool}" ]; then
      key_pool="${key_pool},${!key_var}"
    else
      key_pool="${!key_var}"
    fi
  done

  OPENAI_API_KEYS="${key_pool}" nohup python "$PROJECT/main_param.py" \
    --run_name "$worker" \
    --root_dir "$HE_RUN_ROOT/workers" \
    --dataset_path "$ds_shard" \
    --strategy dot \
    --language py \
    --model qwen2_1.5b \
    --pass_at_k 1 \
    --max_iters 5 \
    --inner_iter 5 \
    --use_mistakes \
    --mistake_json_path "$pf_shard" \
    --phase1_only \
    --phase2_search_mode search \
    --verbose \
    > "$log_file" 2>&1 &

  echo -e "${worker}\t$!" >> "$PIDS_FILE"
done

echo "[ORCH] Phase1 started. Run root: $HE_RUN_ROOT"
echo "[ORCH] PID file: $PIDS_FILE"
echo "[ORCH] After Phase1 completes, run merge and Phase2"
