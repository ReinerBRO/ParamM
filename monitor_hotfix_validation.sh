#!/bin/bash
cd /data/user/user06/ParamAgent/direction3

RUN_NAME="dir3_router_mbpp_transformer_phase2only_hotfix_20260306_110253"
RUN_DIR="results/mbpp_router_runs/$RUN_NAME"

log() {
    echo "[$(date +%H:%M:%S)] $1"
}

log "Monitoring hotfix validation: $RUN_NAME"

while true; do
    if [ -f "$RUN_DIR/results.txt" ]; then
        log "Results found!"
        cat "$RUN_DIR/results.txt"
        
        PHASE2_ACC=$(grep "phase2_acc" "$RUN_DIR/results.txt" | grep -oP '\d+\.\d+' | head -1)
        
        if [ -n "$PHASE2_ACC" ]; then
            log ""
            log "phase2_acc: $PHASE2_ACC%"
            log "Target: 75.31%"
            
            if (( $(echo "$PHASE2_ACC > 75.31" | bc -l) )); then
                log "SUCCESS: Target achieved! ($PHASE2_ACC% > 75.31%)"
                exit 0
            else
                log "Gap: $(echo "75.31 - $PHASE2_ACC" | bc) percentage points"
                exit 1
            fi
        fi
        break
    fi
    
    if [ -d "$RUN_DIR/shards" ]; then
        PHASE2_DONE=$(find "$RUN_DIR/shards" -name "mbpp_pitfalls_shard_*.jsonl" -type f | wc -l)
        log "Progress: phase2=$PHASE2_DONE/24"
    fi
    
    sleep 60
done
