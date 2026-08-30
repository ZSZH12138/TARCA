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

remaining_rental_hours=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --remaining-rental-hours)
      remaining_rental_hours="${2:-}"
      shift 2
      ;;
    *)
      printf 'Unsupported argument\n' >&2
      exit 2
      ;;
  esac
done
if [[ -z "${remaining_rental_hours}" ]]; then
  printf 'Missing --remaining-rental-hours\n' >&2
  exit 2
fi

tarca_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
artifact_root="${TARCA_E01_V2_ARTIFACT_DIR:-${tarca_root}/artifacts/e01-v2-server}"

if ! "${tarca_python}" -c 'import pydantic, psutil, torch, uvicorn, yaml' >/dev/null 2>&1; then
  "${tarca_python}" -m pip install --require-hashes --no-cache-dir \
    -r "${tarca_root}/deploy/stage1b/requirements-server.lock"
fi

export PYTHONPATH="${tarca_root}/deploy/e01/py310:${tarca_root}/src"
export TARCA_EXECUTION_KIND=e01-v2
export TARCA_E01_V2_CONFIG="${tarca_root}/configs/e01/e01_v2.yaml"
export TARCA_E01_V2_ARTIFACT_ROOT="${artifact_root}"
export TARCA_RUNTIME_DATABASE="${artifact_root}/runtime/execution.sqlite3"
export TARCA_RUNTIME_STATIC_ROOT="${tarca_root}/frontend/stage1b-monitor/dist"

mkdir -p "${artifact_root}"
runtime=(
  "${tarca_python}" "${tarca_root}/scripts/run_e01_v2.py"
  --repository-root "${tarca_root}"
  --config "${tarca_root}/configs/e01/e01_v2.yaml"
  --artifact-root "${artifact_root}"
)
"${runtime[@]}" prepare
"${runtime[@]}" dry-run
"${runtime[@]}" preflight --remaining-rental-hours "${remaining_rental_hours}"
