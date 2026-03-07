#!/bin/bash
RUN_NAME="dir3_router_humaneval_transformer_20260306_141201"
RUN_DIR="results/humaneval_router_runs/$RUN_NAME"

log() {
    echo "[$(date +%H:%M:%S)] $1"
}

log "Monitoring HumanEval validation: $RUN_NAME"

while true; do
    if [ -f "$RUN_DIR/results.txt" ]; then
        log "Results found!"
        cat "$RUN_DIR/results.txt"
        
        PHASE2_ACC=$(grep "phase2_acc" "$RUN_DIR/results.txt" | grep -oP '\d+\.\d+' | head -1)
        
        if [ -n "$PHASE2_ACC" ]; then
            log ""
            log "HumanEval phase2_acc: $PHASE2_ACC%"
            log "Previous (old router): 85.98%"
        fi
        break
    fi
    
    if [ -d "$RUN_DIR/shards" ]; then
        PHASE1_DONE=$(find "$RUN_DIR/shards" -name "humaneval_shard_*.jsonl" -type f 2>/dev/null | wc -l)
        PHASE2_DONE=$(find "$RUN_DIR/shards" -name "humaneval_pitfalls_shard_*.jsonl" -type f 2>/dev/null | wc -l)
        log "Progress: phase1=$PHASE1_DONE/24, phase2=$PHASE2_DONE/24"
    fi
    
    sleep 60
done
