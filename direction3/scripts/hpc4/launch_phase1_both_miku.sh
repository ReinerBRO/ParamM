#!/usr/bin/env bash
set -euo pipefail

cd "/data/user/user06/ParamAgent/direction3"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /data/user/user06/miniconda3/etc/profile.d/conda.sh
conda activate paramm

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

if [ -f .env ]; then
  set -a
  source ./.env
  set +a
fi

if [ -f .yunwu ]; then
  idx=2
  while IFS= read -r line; do
    key="${line//[$'\r\n\t ']}"
    if [ -n "$key" ]; then
      export "ZZZ_API_KEY_${idx}=$key"
      idx=$((idx+1))
    fi
  done < ./.yunwu
fi

for i in $(seq 2 25); do
  var="ZZZ_API_KEY_${i}"
  if [ -z "${!var:-}" ]; then
    echo "[ERROR] missing ${var}" >&2
    exit 1
  fi
done

mkdir -p scripts/hpc4 results/humaneval/paramAgent results/mbpp/paramAgent
cp "/data/user/user06/ParamAgent/scripts/hpc4/run_phase1_sharded_miku.sh" scripts/hpc4/
cp "/data/user/user06/ParamAgent/scripts/hpc4/run_phase2_sharded_miku.sh" scripts/hpc4/
cp "/data/user/user06/ParamAgent/scripts/hpc4/run_mbpp_phase1_sharded_miku.sh" scripts/hpc4/
cp "/data/user/user06/ParamAgent/scripts/hpc4/run_mbpp_phase2_sharded_miku.sh" scripts/hpc4/
cp "/data/user/user06/ParamAgent/scripts/hpc4/merge_phase1_shards.py" scripts/hpc4/
cp "/data/user/user06/ParamAgent/scripts/hpc4/merge_phase2_shards.py" scripts/hpc4/

# Avoid dependency on openai package for visible tests generation.
python - <<'PY'
import json
from pathlib import Path
src = Path('benchmarks/humaneval_full.jsonl')
dst = Path('benchmarks/humaneval_visible_tests.jsonl')
rows = []
with src.open('r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        rows.append({
            'task_id': obj.get('task_id'),
            'entry_point': obj.get('entry_point'),
            'given_tests': []
        })
with dst.open('w', encoding='utf-8') as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print(f'[INFO] wrote {len(rows)} rows to {dst}')
PY

RUN_TS="$(date +%Y%m%d_%H%M%S)"
HE_RUN="dir3_router_humaneval_${RUN_TS}"
MBPP_RUN="dir3_router_mbpp_${RUN_TS}"

echo "HE_RUN=${HE_RUN}"
echo "MBPP_RUN=${MBPP_RUN}"

export NUM_WORKERS=24
PITFALL_PATH="benchmarks/code_pitfalls/humaneval_full_pitfalls_lora_20260302_232456_clean.jsonl" \
  bash scripts/hpc4/run_phase1_sharded_miku.sh paramm "${HE_RUN}"

PITFALL_PATH="benchmarks/code_pitfalls/mbpp_pitfalls_lora_20260303_021714.jsonl" \
  bash scripts/hpc4/run_mbpp_phase1_sharded_miku.sh paramm "${MBPP_RUN}"

echo "[INFO] Phase1 submitted for both datasets"
echo "[INFO] humaneval run root: results/humaneval/paramAgent/${HE_RUN}"
echo "[INFO] mbpp run root: results/mbpp/paramAgent/${MBPP_RUN}"
