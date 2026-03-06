#!/bin/bash
# Continuous supervision loop for Direction 3 pipeline

cd /data/user/user06/ParamAgent/direction3

LOG_FILE="supervisor.log"
TRAIN_PID=1361157
AUTO_PID=1361751

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=== Continuous Supervisor Started ==="
log "Training PID: $TRAIN_PID"
log "Auto-integration PID: $AUTO_PID"

# Phase 1: Monitor training
log "Phase 1: Monitoring training..."
while kill -0 $TRAIN_PID 2>/dev/null; do
    # Check progress
    LAST_EPOCH=$(tail -100 router_training.log | grep "Epoch" | tail -1)
    LAST_BEST=$(tail -100 router_training.log | grep "best model" | tail -1)
    
    if [ -n "$LAST_EPOCH" ]; then
        log "Training: $LAST_EPOCH"
    fi
    if [ -n "$LAST_BEST" ]; then
        log "  $LAST_BEST"
    fi
    
    sleep 60
done

log "Training completed!"

# Phase 2: Verify checkpoint and integration
log "Phase 2: Verifying checkpoint and integration..."
sleep 5

CKPT_PATH="router_checkpoints/transformer_mbpp_v1/router.ckpt"
if [ -f "$CKPT_PATH" ]; then
    SIZE=$(du -h "$CKPT_PATH" | cut -f1)
    log "Checkpoint found: $SIZE"
else
    log "ERROR: Checkpoint not found!"
    exit 1
fi

# Wait for auto-integration to complete
log "Waiting for auto-integration..."
while kill -0 $AUTO_PID 2>/dev/null; do
    sleep 10
done

log "Auto-integration completed!"

# Check if validation job was submitted
sleep 5
JOBS=$(squeue -u user06 -h | wc -l)
log "SLURM jobs active: $JOBS"

if [ $JOBS -gt 0 ]; then
    log "Validation job submitted successfully!"
    log "Phase 3: Monitoring validation run..."
    
    # Monitor validation job
    while [ $(squeue -u user06 -h | wc -l) -gt 0 ]; do
        JOBS_STATUS=$(squeue -u user06 -h | head -3)
        log "Jobs running:"
        echo "$JOBS_STATUS" | while read line; do
            log "  $line"
        done
        sleep 120
    done
    
    log "Validation jobs completed!"
    log "Phase 4: Checking results..."
    
    # Find latest results
    LATEST_RESULT=$(find results/mbpp_router_runs -name "results.txt" -type f -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -n "$LATEST_RESULT" ]; then
        log "Latest results: $LATEST_RESULT"
        
        # Extract phase2_acc
        PHASE2_ACC=$(grep "phase2_acc" "$LATEST_RESULT" | grep -oP '\d+\.\d+' | head -1)
        
        if [ -n "$PHASE2_ACC" ]; then
            log "phase2_acc: $PHASE2_ACC%"
            
            # Check if target achieved
            if (( $(echo "$PHASE2_ACC > 75.31" | bc -l) )); then
                log "SUCCESS: Target achieved! ($PHASE2_ACC% > 75.31%)"
                exit 0
            else
                log "Target not达标: $PHASE2_ACC% <= 75.31%"
                log "Architectural iteration needed - delegate to Codex"
                exit 2
            fi
        else
            log "Could not extract phase2_acc from results"
            exit 3
        fi
    else
        log "No results found"
        exit 4
    fi
else
    log "ERROR: No validation job submitted"
    exit 5
fi
