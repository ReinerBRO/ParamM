#!/bin/bash

while true; do
    clear
    echo "=== Direction 3 Pipeline Status ==="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Training status
    if ps -p 1353225 > /dev/null 2>&1; then
        ELAPSED=$(ps -p 1353225 -o etime= | tr -d ' ')
        echo "Training (PID 1353225): RUNNING (elapsed: $ELAPSED)"
    else
        echo "Training (PID 1353225): COMPLETED"
    fi
    
    # Monitor status
    if ps -p 1359279 > /dev/null 2>&1; then
        echo "Monitor (PID 1359279): RUNNING"
    else
        echo "Monitor (PID 1359279): STOPPED"
    fi
    
    # Check for checkpoint
    if [ -f "router_checkpoints/transformer_mbpp_v1/router.ckpt" ]; then
        SIZE=$(du -h router_checkpoints/transformer_mbpp_v1/router.ckpt | cut -f1)
        echo "Checkpoint: EXISTS ($SIZE)"
    else
        echo "Checkpoint: NOT YET"
    fi
    
    # Monitor log
    if [ -f "monitor_launch.log" ]; then
        echo ""
        echo "--- Monitor Log (last 10 lines) ---"
        tail -10 monitor_launch.log
    fi
    
    # Check for validation job
    echo ""
    echo "--- SLURM Jobs ---"
    squeue -u user06 | head -20
    
    sleep 60
done
