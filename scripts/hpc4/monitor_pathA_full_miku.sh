#!/usr/bin/env bash
set -uo pipefail

RUN_NAME="${1:-paramAgent_humaneval_llama8b_miku}"
RUN_DIR="/data/user/user06/ParamAgent/results/humaneval/paramAgent/${RUN_NAME}"
NOHUP_LOG="/data/user/user06/ParamAgent/results/humaneval/paramAgent/${RUN_NAME}_nohup.log"
MONITOR_LOG="/tmp/${RUN_NAME}_monitor.log"
ACCEPT_JSON="/tmp/${RUN_NAME}_acceptance.json"
POLL_SEC="${POLL_SEC:-60}"
DEAD_THRESHOLD="${DEAD_THRESHOLD:-3}"

echo "[monitor] start $(date '+%F %T')" | tee -a "$MONITOR_LOG"
echo "[monitor] run_dir=${RUN_DIR}" | tee -a "$MONITOR_LOG"

dead_count=0
while true; do
  pid_line="$(ssh -o ConnectTimeout=8 -o BatchMode=yes miku "ps -eo pid,args | grep 'main_param.py' | grep '${RUN_NAME}' | grep -v grep" 2>/dev/null || true)"
  alive="no"
  pid="NA"
  if [ -n "$pid_line" ]; then
    alive="yes"
    pid="$(echo "$pid_line" | awk '{print $1}' | head -n1)"
    dead_count=0
  else
    dead_count=$((dead_count + 1))
  fi

  first_count=0
  second_count=0
  if [ -f "${RUN_DIR}/first_stage_log.jsonl" ]; then
    first_count="$(wc -l < "${RUN_DIR}/first_stage_log.jsonl" | tr -d ' ')"
  fi
  if [ -f "${RUN_DIR}/second_stage_log.jsonl" ]; then
    second_count="$(wc -l < "${RUN_DIR}/second_stage_log.jsonl" | tr -d ' ')"
  fi

  echo "[monitor] $(date '+%F %T') alive=${alive} pid=${pid} first=${first_count} second=${second_count}" | tee -a "$MONITOR_LOG"

  if [ "$dead_count" -ge "$DEAD_THRESHOLD" ]; then
    echo "[monitor] detected process exit after ${dead_count} consecutive dead checks" | tee -a "$MONITOR_LOG"
    break
  fi
  sleep "$POLL_SEC"
done

echo "[monitor] process ended, evaluating acceptance..." | tee -a "$MONITOR_LOG"
python /data/user/user06/ParamAgent/scripts/hpc4/evaluate_pathA_acceptance.py "$RUN_DIR" "$NOHUP_LOG" | tee "$ACCEPT_JSON" || true
echo "[monitor] done $(date '+%F %T')" | tee -a "$MONITOR_LOG"
