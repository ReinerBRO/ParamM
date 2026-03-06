#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/user/user06/ParamAgent/direction3"
RUN_SCRIPT="${PROJECT_ROOT}/scripts/hpc4/run_mbpp_router_fixed_pipeline.sh"
STATE_DIR="${PROJECT_ROOT}/results/mbpp_router_runs"
WATCH_LOG="${STATE_DIR}/api_watchdog.log"
WATCH_PID="${STATE_DIR}/api_watchdog.pid"
POLL_SECONDS="${POLL_SECONDS:-1800}"

mkdir -p "${STATE_DIR}"

log() {
  printf '%s [WATCHDOG] %s\n' "$(date '+%F %T')" "$1" | tee -a "${WATCH_LOG}"
}

if [ -f "${WATCH_PID}" ]; then
  old_pid="$(cat "${WATCH_PID}" 2>/dev/null || true)"
  if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
    log "already running pid=${old_pid}"
    exit 0
  fi
fi

echo "$$" > "${WATCH_PID}"
cleanup() {
  rm -f "${WATCH_PID}"
}
trap cleanup EXIT

check_relay_ready() {
  set +e
  python - <<'PY'
import os
import sys
from pathlib import Path
import requests

env_path = Path('/data/user/user06/ParamAgent/direction3/.env')
if not env_path.exists():
    print('env_missing')
    sys.exit(2)

env_map = {}
for raw in env_path.read_text(encoding='utf-8', errors='ignore').splitlines():
    line = raw.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    env_map[k.strip()] = v.strip()

base_url = env_map.get('OPENAI_BASE_URL', 'https://api.zhizengzeng.com/v1').rstrip('/')
api_key = env_map.get('ZZZ_API_KEY_2', '')
if not api_key:
    print('key_missing')
    sys.exit(2)

for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY'):
    os.environ.pop(k, None)

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
}

try:
    chat_resp = requests.post(
        f'{base_url}/chat/completions',
        headers=headers,
        json={
            'model': 'llama3_1_8b',
            'messages': [{'role': 'user', 'content': 'ping'}],
            'max_tokens': 8,
            'temperature': 0,
        },
        timeout=25,
    )
except Exception as exc:
    print(f'chat_exc={type(exc).__name__}')
    sys.exit(1)

try:
    emb_resp = requests.post(
        f'{base_url}/embeddings',
        headers=headers,
        json={
            'model': 'text-embedding-3-small',
            'input': 'ping',
        },
        timeout=25,
    )
except Exception as exc:
    print(f'emb_exc={type(exc).__name__}')
    sys.exit(1)

print(f'chat={chat_resp.status_code} emb={emb_resp.status_code}')
if chat_resp.status_code == 200 and emb_resp.status_code == 200:
    sys.exit(0)
sys.exit(1)
PY
  rc=$?
  set -e
  return "$rc"
}

log "started poll_seconds=${POLL_SECONDS}"

while true; do
  if check_relay_ready; then
    run_name="dir3_router_mbpp_fix_auto_$(date +%Y%m%d_%H%M%S)"
    log "relay healthy; launching MBPP run ${run_name}"
    nohup bash "${RUN_SCRIPT}" "${run_name}" >> "${STATE_DIR}/api_watchdog_launch.log" 2>&1 &
    launcher_pid=$!
    echo "${launcher_pid}" > "${STATE_DIR}/latest_pipeline_launcher.pid"
    log "pipeline launcher pid=${launcher_pid}"
    exit 0
  fi

  log "relay not ready; retry after ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done
