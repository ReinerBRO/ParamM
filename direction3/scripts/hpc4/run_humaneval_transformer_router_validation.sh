#!/bin/bash
# HumanEval validation with Transformer router + hotfix

set -e

PROJECT_ROOT="/data/user/user06/ParamAgent/direction3"
cd "$PROJECT_ROOT"

# Configuration
RUN_NAME="${RUN_NAME:-dir3_router_humaneval_transformer_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-24}"
CONDA_ENV="${CONDA_ENV:-paramm}"
ROUTER_CONF_THRESHOLD="${ROUTER_CONF_THRESHOLD:-0.6}"
ROUTER_CKPT="${ROUTER_CKPT:-${PROJECT_ROOT}/router_checkpoints/transformer_mbpp_v1/router.ckpt}"

# Paths
DATASET="${PROJECT_ROOT}/benchmarks/humaneval_full.jsonl"
PITFALLS="${PROJECT_ROOT}/benchmarks/code_pitfalls/humaneval_full_pitfalls_lora_20260302_232456_clean.jsonl"
RUN_ROOT="${PROJECT_ROOT}/results/humaneval_router_runs/${RUN_NAME}"

echo "[VALIDATION] run_name=$RUN_NAME"
echo "[VALIDATION] router_ckpt=$ROUTER_CKPT"
echo "[VALIDATION] router_conf_threshold=$ROUTER_CONF_THRESHOLD"
echo "[VALIDATION] num_workers=$NUM_WORKERS"

# Verify checkpoint exists
if [ ! -f "$ROUTER_CKPT" ]; then
    echo "ERROR: Router checkpoint not found: $ROUTER_CKPT"
    exit 1
fi

# Create run directory
mkdir -p "$RUN_ROOT"/{shards,logs,workers}

# Export environment
export RELAY_FALLBACK_MODEL=""
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "[PIPELINE] run_name=$RUN_NAME"
echo "[PIPELINE] run_root=$RUN_ROOT"
echo "[PIPELINE] dataset=$DATASET"
echo "[PIPELINE] pitfalls=$PITFALLS"
echo "[PIPELINE] router_ckpt=$ROUTER_CKPT"
echo "[PIPELINE] relay_fallback_model=DISABLED"

# Count total items
TOTAL_ITEMS=$(wc -l < "$DATASET")
echo "sharded $TOTAL_ITEMS items across $NUM_WORKERS workers"

# Phase 1: Initial generation
echo "[PIPELINE] phase1 launched"

# Pre-shard the dataset
python - <<PY "$DATASET" "$PITFALLS" "$RUN_ROOT/shards" "$NUM_WORKERS"
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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
    ds_out = os.path.join(out_dir, f"humaneval_shard_{w:02d}.jsonl")
    pf_out = os.path.join(out_dir, f"humaneval_pitfalls_shard_{w:02d}.jsonl")
    write_jsonl(ds_out, dataset[start:end])
    write_jsonl(pf_out, pitfalls[start:end])
    print(f"worker_{w:02d}: {start}:{end} -> {end-start} items")
PY

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    WORKER_ID=$(printf "%02d" $i)
    DS_SHARD="$RUN_ROOT/shards/humaneval_shard_${WORKER_ID}.jsonl"
    PF_SHARD="$RUN_ROOT/shards/humaneval_pitfalls_shard_${WORKER_ID}.jsonl"

    python main_param.py \
        --run_name "worker_${WORKER_ID}" \
        --root_dir "$RUN_ROOT/workers" \
        --dataset_path "$DS_SHARD" \
        --strategy dot \
        --language py \
        --model llama3_1_8b \
        --pass_at_k 1 \
        --max_iters 5 \
        --inner_iter 5 \
        --use_mistakes \
        --mistake_json_path "$PF_SHARD" \
        --phase1_only \
        > "$RUN_ROOT/logs/worker_${WORKER_ID}_phase1.log" 2>&1 &

    echo $! >> "$RUN_ROOT/phase1_worker_pids.tsv"
done

# Monitor phase1
while true; do
    ALIVE=$(ps -p $(cat "$RUN_ROOT/phase1_worker_pids.tsv" 2>/dev/null | tr '\n' ',' | sed 's/,$//') 2>/dev/null | grep -c python || echo 0)
    echo "[PIPELINE] phase1_alive=$ALIVE"
    [ $ALIVE -eq 0 ] && break
    sleep 30
done

# Merge phase1
echo "[PIPELINE] phase1 merge started"
python scripts/merge_shards.py \
    --shard_pattern "$RUN_ROOT/shards/humaneval_shard_*.jsonl" \
    --output_dir "$RUN_ROOT/merged_phase1" \
    --phase phase1

echo "[PIPELINE] phase1 merge finished (rc=$?)"

# Phase 2: Router-guided retry with memory
echo "[PIPELINE] phase2 launched (router ON)"

for i in $(seq 0 $((NUM_WORKERS - 1))); do
    WORKER_ID=$(printf "%02d" $i)
    DS_SHARD="$RUN_ROOT/shards/humaneval_shard_${WORKER_ID}.jsonl"
    PF_SHARD="$RUN_ROOT/shards/humaneval_pitfalls_shard_${WORKER_ID}.jsonl"

    python main_param.py \
        --run_name "worker_${WORKER_ID}" \
        --root_dir "$RUN_ROOT/workers" \
        --dataset_path "$DS_SHARD" \
        --strategy dot \
        --language py \
        --model llama3_1_8b \
        --pass_at_k 1 \
        --max_iters 5 \
        --inner_iter 5 \
        --use_mistakes \
        --mistake_json_path "$PF_SHARD" \
        --global_mem_bank_path "$RUN_ROOT/merged_phase1/mem_bank.pkl" \
        --router_enable \
        --router_ckpt_path "$ROUTER_CKPT" \
        --router_conf_threshold "$ROUTER_CONF_THRESHOLD" \
        > "$RUN_ROOT/logs/worker_${WORKER_ID}_phase2.log" 2>&1 &

    echo $! >> "$RUN_ROOT/phase2_worker_pids.tsv"
done

# Monitor phase2
while true; do
    ALIVE=$(ps -p $(cat "$RUN_ROOT/phase2_worker_pids.tsv" 2>/dev/null | tr '\n' ',' | sed 's/,$//') 2>/dev/null | grep -c python || echo 0)
    echo "[PIPELINE] phase2_alive=$ALIVE"
    [ $ALIVE -eq 0 ] && break
    sleep 30
done

# Merge phase2
echo "[PIPELINE] phase2 merge started"
timeout 7200 python scripts/merge_shards.py \
    --shard_pattern "$RUN_ROOT/shards/humaneval_pitfalls_shard_*.jsonl" \
    --output_dir "$RUN_ROOT/merged_phase2" \
    --phase phase2 \
    --phase1_dir "$RUN_ROOT/merged_phase1"

MERGE_RC=$?
echo "[PIPELINE] phase2 merge finished (rc=$MERGE_RC)"

# Generate results
python scripts/compute_results.py \
    --phase1_dir "$RUN_ROOT/merged_phase1" \
    --phase2_dir "$RUN_ROOT/merged_phase2" \
    --output "$RUN_ROOT/results.txt"

echo "[PIPELINE] done"
echo "[VALIDATION] results: $RUN_ROOT/results.txt"

cat "$RUN_ROOT/results.txt"
