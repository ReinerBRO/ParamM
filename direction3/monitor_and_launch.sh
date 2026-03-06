#!/bin/bash
set -e

cd /data/user/user06/ParamAgent/direction3

PID=1353225
CKPT_PATH="router_checkpoints/transformer_mbpp_v1/router.ckpt"

echo "[$(date +%H:%M:%S)] Monitoring training PID $PID..."

# Wait for training to complete
while kill -0 $PID 2>/dev/null; do
    sleep 120  # Check every 2 minutes
    echo "[$(date +%H:%M:%S)] Training still running..."
done

echo "[$(date +%H:%M:%S)] Training completed!"

# Verify checkpoint
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

CKPT_SIZE=$(du -h "$CKPT_PATH" | cut -f1)
echo "Checkpoint found: $CKPT_PATH ($CKPT_SIZE)"

# Backup current paramAgent.py
cp paramAgent.py paramAgent.py.backup_$(date +%Y%m%d_%H%M%S)
echo "Backed up paramAgent.py"

# Integrate router
echo "[$(date +%H:%M:%S)] Integrating Transformer router..."
python integrate_router.py
if [ $? -ne 0 ]; then
    echo "ERROR: Router integration failed"
    exit 1
fi

echo "[$(date +%H:%M:%S)] Router integrated successfully!"

# Launch 24-worker MBPP validation
echo "[$(date +%H:%M:%S)] Launching 24-worker MBPP validation run..."

cd scripts/hpc4
sbatch run_mbpp_router_fixed_pipeline.sh

echo "[$(date +%H:%M:%S)] Validation job submitted!"
echo "Monitor with: squeue -u user06"
