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

case "${1:-}" in
  launch|resume)
    command="$1"
    shift
    ;;
  *)
    printf 'Usage: %s {launch|resume} --acknowledgement <exact-v2-token>\n' "$0" >&2
    exit 2
    ;;
esac

tarca_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_root="${TARCA_E01_V2_ARTIFACT_DIR:-${tarca_root}/artifacts/e01-v2-server}"
runtime_dir="${artifact_root}/runtime"
mkdir -p "${runtime_dir}"

export PYTHONPATH="${tarca_root}/deploy/e01/py310:${tarca_root}/src"
export TARCA_EXECUTION_KIND=e01-v2
export TARCA_E01_V2_CONFIG="${tarca_root}/configs/e01/e01_v2.yaml"
export TARCA_E01_V2_ARTIFACT_ROOT="${artifact_root}"
export TARCA_RUNTIME_DATABASE="${runtime_dir}/execution.sqlite3"
export TARCA_RUNTIME_STATIC_ROOT="${tarca_root}/frontend/stage1b-monitor/dist"
export TARCA_MONITOR_BIND_HOST="${TARCA_MONITOR_BIND_HOST:-127.0.0.1}"

nohup "${tarca_python}" "${tarca_root}/scripts/run_e01_v2.py" \
  --repository-root "${tarca_root}" \
  --config "${tarca_root}/configs/e01/e01_v2.yaml" \
  --artifact-root "${artifact_root}" \
  "${command}" "$@" >"${runtime_dir}/runtime.log" 2>&1 &
runtime_pid="$!"
printf '%s\n' "${runtime_pid}" >"${runtime_dir}/runtime.pid"

for _ in $(seq 1 100); do
  [[ -f "${TARCA_RUNTIME_DATABASE}" ]] && break
  kill -0 "${runtime_pid}" 2>/dev/null || break
  sleep 0.1
done
if [[ ! -f "${TARCA_RUNTIME_DATABASE}" ]] || ! kill -0 "${runtime_pid}" 2>/dev/null; then
  printf 'E01-v2 runtime did not become stable; inspect runtime.log\n' >&2
  exit 5
fi

nohup "${tarca_python}" -m uvicorn tarca.monitoring.server:create_app_from_environment \
  --factory --host "${TARCA_MONITOR_BIND_HOST}" --port 8765 --no-access-log \
  >"${runtime_dir}/monitor.log" 2>&1 &
monitor_pid="$!"
printf '%s\n' "${monitor_pid}" >"${runtime_dir}/monitor.pid"
monitor_ready="false"
for _ in $(seq 1 40); do
  if ! kill -0 "${monitor_pid}" 2>/dev/null; then
    break
  fi
  if "${tarca_python}" -c \
    'from urllib.request import urlopen; urlopen("http://127.0.0.1:8765/api/v1/run", timeout=1).read()' \
    >/dev/null 2>&1; then
    monitor_ready="true"
    break
  fi
  sleep 0.25
done
if [[ "${monitor_ready}" != "true" ]]; then
  kill -TERM "${runtime_pid}" 2>/dev/null || true
  kill -TERM "${monitor_pid}" 2>/dev/null || true
  wait "${runtime_pid}" 2>/dev/null || true
  printf 'E01-v2 monitor did not become stable; formal runtime was stopped\n' >&2
  exit 6
fi

printf 'E01-v2 runtime and read-only monitor are stable\n'
printf 'Monitor endpoint: http://127.0.0.1:8765\n'
