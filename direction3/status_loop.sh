#!/bin/bash

while true; do
    clear
    echo "=== Direction 3 Transformer Router Pipeline ==="
    echo "Time: $(date '+%H:%M:%S')"
    echo ""
    
    # Training
    if ps -p 1361157 > /dev/null 2>&1; then
        ELAPSED=$(ps -p 1361157 -o etime= | tr -d ' ')
        echo "Training (PID 1361157): RUNNING (elapsed: $ELAPSED)"
    else
        echo "Training (PID 1361157): COMPLETED"
    fi
    
    # Auto-integration
    if ps -p 1361751 > /dev/null 2>&1; then
        echo "Auto-integration (PID 1361751): RUNNING"
    else
        echo "Auto-integration (PID 1361751): STOPPED"
    fi
    
    # Training progress
    if [ -f "router_training.log" ]; then
        echo ""
        echo "--- Training Progress ---"
        tail -15 router_training.log | grep -E "(Epoch|loss=|best model)" | tail -5
    fi
    
    # Checkpoint
    if [ -f "router_checkpoints/transformer_mbpp_v1/router.ckpt" ]; then
        SIZE=$(du -h router_checkpoints/transformer_mbpp_v1/router.ckpt | cut -f1)
        echo ""
        echo "Checkpoint: $SIZE"
    fi
    
    # SLURM jobs
    echo ""
    echo "--- SLURM Jobs ---"
    squeue -u user06 2>/dev/null | head -5 || echo "No jobs"
    
    sleep 180  # Update every 3 minutes
done
