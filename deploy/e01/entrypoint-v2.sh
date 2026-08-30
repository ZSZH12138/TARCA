#!/usr/bin/env bash
set -euo pipefail

if [[ -x "/opt/conda/bin/python" ]]; then
  tarca_python="/opt/conda/bin/python"
elif command -v python >/dev/null 2>&1; then
  tarca_python="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  tarca_python="$(command -v python3)"
else
  printf 'Python interpreter is missing\n' >&2
  exit 4
fi

runtime_pid=""
monitor_pid=""

terminate() {
  if [[ -n "${runtime_pid}" ]]; then kill -TERM "${runtime_pid}" 2>/dev/null || true; fi
  if [[ -n "${monitor_pid}" ]]; then kill -TERM "${monitor_pid}" 2>/dev/null || true; fi
}

runtime_arguments=(
  --repository-root /opt/tarca
  --config /opt/tarca/configs/e01/e01_v2.yaml
  --artifact-root /opt/tarca/artifacts/e01-v2
)

if [[ "${1:-}" == "launch" || "${1:-}" == "resume" ]]; then
  "${tarca_python}" scripts/run_e01_v2.py "${runtime_arguments[@]}" "$@" &
  runtime_pid="$!"
  for _ in $(seq 1 100); do
    [[ -f "${TARCA_RUNTIME_DATABASE}" ]] && break
    kill -0 "${runtime_pid}" 2>/dev/null || break
    sleep 0.1
  done
  if [[ -f "${TARCA_RUNTIME_DATABASE}" ]]; then
    "${tarca_python}" -m uvicorn tarca.monitoring.server:create_app_from_environment \
      --factory --host "${TARCA_MONITOR_BIND_HOST:-127.0.0.1}" --port 8765 --no-access-log &
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

exec "${tarca_python}" scripts/run_e01_v2.py "${runtime_arguments[@]}" "$@"
