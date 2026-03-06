#!/bin/bash
cd /data/user/user06/ParamAgent/direction3

RUN_NAME="dir3_router_mbpp_transformer_val_20260306_014139"
RUN_DIR="results/mbpp_router_runs/$RUN_NAME"

log() {
    echo "[$(date +%H:%M:%S)] $1"
}

log "Monitoring Transformer router validation: $RUN_NAME"

# Monitor until completion
while true; do
    # Check for results file
    if [ -f "$RUN_DIR/results.txt" ]; then
        log "Results found!"
        cat "$RUN_DIR/results.txt"
        
        # Extract phase2_acc
        PHASE2_ACC=$(grep "phase2_acc" "$RUN_DIR/results.txt" | grep -oP '\d+\.\d+' | head -1)
        
        if [ -n "$PHASE2_ACC" ]; then
            log ""
            log "phase2_acc: $PHASE2_ACC%"
            
            if (( $(echo "$PHASE2_ACC > 75.31" | bc -l) )); then
                log "SUCCESS: Target achieved! ($PHASE2_ACC% > 75.31%)"
                exit 0
            else
                log "Target not达标: $PHASE2_ACC% <= 75.31%"
                log "Gap: $(echo "75.31 - $PHASE2_ACC" | bc) percentage points"
                exit 1
            fi
        fi
        break
    fi
    
    # Check progress
    if [ -d "$RUN_DIR/shards" ]; then
        PHASE1_DONE=$(find "$RUN_DIR/shards" -name "mbpp_shard_*.jsonl" -type f | wc -l)
        PHASE2_DONE=$(find "$RUN_DIR/shards" -name "mbpp_pitfalls_shard_*.jsonl" -type f | wc -l)
        log "Progress: phase1=$PHASE1_DONE/24, phase2=$PHASE2_DONE/24"
    fi
    
    sleep 60
done
