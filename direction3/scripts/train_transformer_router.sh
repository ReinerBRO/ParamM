#!/bin/bash
# Train Transformer-based router for direction3
# This script:
# 1. Builds training dataset from phase1/phase2 logs
# 2. Trains the Transformer router
# 3. Validates the trained model

set -e

# Configuration
PHASE1_LOG="${PHASE1_LOG:-/data/user/user06/ParamAgent/direction3/logs/phase1_mbpp.jsonl}"
PHASE2_LOG="${PHASE2_LOG:-/data/user/user06/ParamAgent/direction3/logs/phase2_mbpp.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/user/user06/ParamAgent/direction3/router_checkpoints/transformer_v1}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-1e-4}"

echo "=== Direction 3 Transformer Router Training ==="
echo "Phase1 log: $PHASE1_LOG"
echo "Phase2 log: $PHASE2_LOG"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Step 1: Build training dataset
echo "[Step 1/3] Building training dataset..."
python -m memory_router.build_transformer_dataset \
    --phase1_jsonl "$PHASE1_LOG" \
    --phase2_jsonl "$PHASE2_LOG" \
    --output_dir "$OUTPUT_DIR/data" \
    --val_ratio 0.2

if [ $? -ne 0 ]; then
    echo "Error: Failed to build dataset"
    exit 1
fi

TRAIN_DATA="$OUTPUT_DIR/data/train.jsonl"
VAL_DATA="$OUTPUT_DIR/data/val.jsonl"

echo ""
echo "[Step 2/3] Training Transformer router..."
python -m memory_router.train_transformer_router \
    --train_jsonl "$TRAIN_DATA" \
    --val_jsonl "$VAL_DATA" \
    --output_ckpt "$OUTPUT_DIR/router.ckpt" \
    --d_model 256 \
    --nhead 4 \
    --num_layers 2 \
    --dropout 0.1 \
    --max_candidates 20 \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay 1e-4 \
    --seed 42

if [ $? -ne 0 ]; then
    echo "Error: Failed to train router"
    exit 1
fi

echo ""
echo "[Step 3/3] Validating trained router..."
# Test inference on a sample
python -m memory_router.infer_transformer_router \
    --ckpt "$OUTPUT_DIR/router.ckpt" \
    --state_json_str '{"prompt": "test", "reflections": [], "test_feedback": [], "attempt_count": 1, "prompt_sims": [0.8, 0.6], "reflection_sims": [0.5, 0.4], "negative_sims": [0.1, 0.2]}'

if [ $? -ne 0 ]; then
    echo "Warning: Inference test failed"
fi

echo ""
echo "=== Training Complete ==="
echo "Checkpoint: $OUTPUT_DIR/router.ckpt"
echo "Metrics: $OUTPUT_DIR/train_metrics.json"
echo ""
echo "To use this router, set:"
echo "  --router_enable"
echo "  --router_ckpt_path $OUTPUT_DIR/router.ckpt"
