#!/usr/bin/env bash
set -euo pipefail

runtime_pid=""
monitor_pid=""

terminate() {
  if [[ -n "${runtime_pid}" ]]; then kill -TERM "${runtime_pid}" 2>/dev/null || true; fi
  if [[ -n "${monitor_pid}" ]]; then kill -TERM "${monitor_pid}" 2>/dev/null || true; fi
}

if [[ "${1:-}" == "launch" || "${1:-}" == "resume" ]]; then
  python scripts/run_stage1b_runtime.py "$@" &
  runtime_pid="$!"
  for _ in $(seq 1 100); do
    [[ -f "${TARCA_STAGE1B_DATABASE}" ]] && break
    kill -0 "${runtime_pid}" 2>/dev/null || break
    sleep 0.1
  done
  if [[ -f "${TARCA_STAGE1B_DATABASE}" ]]; then
    python -m uvicorn tarca.monitoring.server:create_app_from_environment \
      --factory --host 0.0.0.0 --port 8765 --no-access-log &
    monitor_pid="$!"
  fi
  trap terminate TERM INT
  set +e
  wait "${runtime_pid}"
  exit_code="$?"
  set -e
  terminate
  if [[ -n "${monitor_pid}" ]]; then wait "${monitor_pid}" 2>/dev/null || true; fi
  exit "${exit_code}"
fi

exec python scripts/run_stage1b_runtime.py "$@"
