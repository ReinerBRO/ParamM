#!/bin/bash
set -e

cd /data/user/user06/ParamAgent/direction3

TRAIN_PID=1361157
CKPT_PATH="router_checkpoints/transformer_mbpp_v1/router.ckpt"

echo "[$(date +%H:%M:%S)] Waiting for training PID $TRAIN_PID..."

# Wait for training to complete
while kill -0 $TRAIN_PID 2>/dev/null; do
    sleep 120
    echo "[$(date +%H:%M:%S)] Training still running..."
done

echo "[$(date +%H:%M:%S)] Training completed!"

# Wait a bit for file writes to complete
sleep 5

# Verify checkpoint
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

CKPT_SIZE=$(du -h "$CKPT_PATH" | cut -f1)
echo "Checkpoint found: $CKPT_PATH ($CKPT_SIZE)"

# Backup and integrate
cp paramAgent.py paramAgent.py.backup_$(date +%Y%m%d_%H%M%S)

# Update paramAgent.py to use transformer router
python << 'PYEOF'
import re

with open('paramAgent.py', 'r') as f:
    content = f.read()

# Update import
content = re.sub(
    r'from memory_router\.infer_router import infer as infer_router',
    '''try:
    from memory_router.infer_transformer_router import infer_transformer_router
    ROUTER_INFER_AVAILABLE = True
except ImportError:
    infer_transformer_router = None
    ROUTER_INFER_AVAILABLE = False''',
    content
)

# Update router_ckpt_path default in function signature
content = re.sub(
    r'router_ckpt_path: str = ""',
    'router_ckpt_path: str = "router_checkpoints/transformer_mbpp_v1/router.ckpt"',
    content
)

with open('paramAgent.py', 'w') as f:
    f.write(content)

print("Router integration completed!")
PYEOF

echo "[$(date +%H:%M:%S)] Launching 24-worker MBPP validation..."

cd scripts/hpc4
sbatch run_mbpp_router_fixed_pipeline.sh

echo "[$(date +%H:%M:%S)] Validation job submitted!"
squeue -u user06
