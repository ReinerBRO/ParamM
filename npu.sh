#!/usr/bin/env bash
set -euo pipefail

# Default: show both dev nodes once.
# Usage:
#   bash npu.sh
#   bash npu.sh miku
#   bash npu.sh yui
#   bash npu.sh mutsumi
#   bash npu.sh eren
#   bash npu.sh all
#   bash npu.sh yui 5        # refresh every 5 seconds
#   bash npu.sh 5            # backward-compatible: all nodes, refresh 5s

NPU_MAX_CHIPS="${NPU_MAX_CHIPS:-8}"
TARGET_ARG="${1:-all}"
REFRESH_SEC="${2:-0}"
ALL_TARGETS=(miku yui mutsumi eren)

# Backward compatibility: `bash npu.sh 5` -> all nodes with refresh.
if [[ "${TARGET_ARG}" =~ ^[0-9]+$ ]]; then
  REFRESH_SEC="${TARGET_ARG}"
  TARGET_ARG="all"
fi

# Accept `mutsumi.sh` / `eren.sh` style target arguments for convenience.
TARGET_ARG="${TARGET_ARG%.sh}"

if ! [[ "${REFRESH_SEC}" =~ ^[0-9]+$ ]]; then
  echo "[npu.sh] invalid refresh seconds: ${REFRESH_SEC}" >&2
  exit 1
fi

TARGETS=()
case "${TARGET_ARG}" in
  all|"")
    TARGETS=("${ALL_TARGETS[@]}")
    ;;
  miku|yui|mutsumi|eren)
    TARGETS=("${TARGET_ARG}")
    ;;
  *)
    echo "[npu.sh] usage: bash npu.sh [all|miku|yui|mutsumi|eren] [refresh_sec]" >&2
    exit 1
    ;;
esac

run_snapshot_target() {
  local target="$1"
  ssh "${target}" "bash -s" -- "${NPU_MAX_CHIPS}" "${target}" <<'EOS'
set -euo pipefail
MAX_CHIPS="${1:-8}"
TARGET_ALIAS="${2:-unknown}"

export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:${LD_LIBRARY_PATH:-}"

mapfile -t PAIRS < <(
  npu-smi info -l | awk -F: -v max_chips="${MAX_CHIPS}" '
    /NPU ID/ {
      gsub(/ /, "", $2);
      npu_id = $2;
    }
    /Chip Count/ {
      gsub(/ /, "", $2);
      chip_cnt = $2 + 0;
      for (i = 0; i < chip_cnt; i++) {
        if (printed < max_chips) {
          print npu_id " " i;
          printed++;
        }
      }
    }
  '
)

echo "[$(date '+%F %T')] target=${TARGET_ALIAS} host=$(hostname) chips=${#PAIRS[@]}"

idx=0
for pair in "${PAIRS[@]}"; do
  npu_id="${pair%% *}"
  chip_id="${pair##* }"
  out="$(timeout 8s npu-smi info -t usages -i "${npu_id}" -c "${chip_id}" 2>/dev/null || true)"

  cap="$(awk -F: '/HBM Capacity\(MB\)/{gsub(/ /,"",$2); print $2; exit}' <<< "${out}")"
  pct="$(awk -F: '/HBM Usage Rate\(%\)/{gsub(/ /,"",$2); print $2; exit}' <<< "${out}")"
  npu_util="$(awk -F: '/NPU Utilization\(%\)/{gsub(/ /,"",$2); print $2; exit}' <<< "${out}")"
  aicore="$(awk -F: '/Aicore Usage Rate\(%\)/{gsub(/ /,"",$2); print $2; exit}' <<< "${out}")"

  used="NA"
  if [[ -n "${cap}" && -n "${pct}" ]]; then
    used="$(awk -v c="${cap}" -v p="${pct}" 'BEGIN { printf "%.0f", c * p / 100.0 }')"
  fi

  printf "chip%02d npu%s_chip%s hbm=%s%% used=%sMB/%sMB npu_util=%s%% aicore=%s%%\n" \
    "${idx}" "${npu_id}" "${chip_id}" "${pct:-NA}" "${used}" "${cap:-NA}" "${npu_util:-NA}" "${aicore:-NA}"
  idx=$((idx + 1))
done
EOS
}

run_snapshot() {
  local rc=0
  for target in "${TARGETS[@]}"; do
    echo "===== ${target} ====="
    if ! run_snapshot_target "${target}"; then
      echo "[npu.sh] WARN: failed to query ${target}" >&2
      rc=1
    fi
    echo
  done
  return "${rc}"
}

if [[ "${REFRESH_SEC}" -gt 0 ]]; then
  while true; do
    run_snapshot
    sleep "${REFRESH_SEC}"
  done
else
  run_snapshot
fi
