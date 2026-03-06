#!/bin/bash
set -e

PID=1353225
CKPT_PATH="router_checkpoints/transformer_mbpp_v1/router.ckpt"

echo "[$(date +%H:%M:%S)] Waiting for training to complete (PID $PID)..."

# Wait for process to finish
while kill -0 $PID 2>/dev/null; do
    sleep 60
done

echo "[$(date +%H:%M:%S)] Training completed!"

# Verify checkpoint exists
if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

CKPT_SIZE=$(du -h "$CKPT_PATH" | cut -f1)
echo "Checkpoint found: $CKPT_PATH ($CKPT_SIZE)"

# Integrate router into paramAgent.py
echo "[$(date +%H:%M:%S)] Integrating Transformer router into paramAgent.py..."

# Backup current version
cp paramAgent.py paramAgent.py.backup_$(date +%Y%m%d_%H%M%S)

# Update router loading logic
python << 'PYTHON_EOF'
import re

with open('paramAgent.py', 'r') as f:
    content = f.read()

# Find the router initialization section and update it
# Look for the router loading code around line 100-200
pattern = r'(# Load router.*?)(router_ckpt_path = .*?\n)(.*?)(self\.router = .*?\n)'

replacement = r'''\1router_ckpt_path = "router_checkpoints/transformer_mbpp_v1/router.ckpt"
    if not os.path.exists(router_ckpt_path):
        router_ckpt_path = "direction3/router_checkpoints/transformer_mbpp_v1/router.ckpt"
    
    print(f"Loading Transformer router from: {router_ckpt_path}")
    \3self.router = load_transformer_router(router_ckpt_path, device=self.device)
'''

content_new = re.sub(pattern, replacement, content, flags=re.DOTALL)

if content_new != content:
    with open('paramAgent.py', 'w') as f:
        f.write(content_new)
    print("Router integration completed!")
else:
    print("WARNING: Could not find router initialization pattern")
PYTHON_EOF

echo "[$(date +%H:%M:%S)] Integration complete. Ready to launch validation run."
echo "Next: Launch 24-worker MBPP pipeline with new router"
