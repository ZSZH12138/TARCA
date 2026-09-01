#!/usr/bin/env bash
set -euo pipefail
umask 027

if [[ "${1:-}" == "--help" ]]; then
  printf '%s\n' "TARCA Stage 2 container-native recovery"
  printf '%s\n' "Restores, preflights, repairs, starts the read-only monitor, and stops before resume."
  exit 0
fi

current_stage="validate-inputs"
report_failure_stage() {
  local exit_code=$?
  if ((exit_code != 0)); then
    printf 'TARCA_DIRECT_BOOTSTRAP_FAILED_STAGE=%s\n' "$current_stage" >&2
  fi
}
trap report_failure_stage EXIT

recovery_archive=""
server_bundle=""
remaining_hours=""
repository_root="/opt/tarca"

while (($#)); do
  case "$1" in
    --recovery-archive) recovery_archive="${2:?missing recovery archive}"; shift 2 ;;
    --server-bundle) server_bundle="${2:?missing server bundle}"; shift 2 ;;
    --remaining-rental-hours) remaining_hours="${2:?missing remaining hours}"; shift 2 ;;
    --repository-root) repository_root="${2:?missing repository root}"; shift 2 ;;
    *) echo "unsupported direct recovery argument" >&2; exit 64 ;;
  esac
done

[[ -n "$recovery_archive" && -n "$server_bundle" ]] || {
  echo "recovery archive and server bundle are required" >&2
  exit 64
}
[[ "$remaining_hours" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "valid remaining rental hours are required" >&2
  exit 64
}

recovery_archive="$(realpath -e -- "$recovery_archive" 2>/dev/null)"
server_bundle="$(realpath -e -- "$server_bundle" 2>/dev/null)"
repository_root="$(realpath -e -- "$repository_root" 2>/dev/null)"
[[ -f "$recovery_archive" && -f "$server_bundle" ]] || {
  echo "recovery kit input is not a regular file" >&2
  exit 66
}

base_python=""
for python_candidate in python python3 /opt/conda/bin/python; do
  if command -v "$python_candidate" >/dev/null 2>&1 || [[ -x "$python_candidate" ]]; then
    if "$python_candidate" -c \
      'import sys, torch; assert sys.version_info[:2] == (3, 10); assert torch.__version__.split("+")[0] == "2.2.2"' \
      >/dev/null 2>&1; then
      base_python="$python_candidate"
      break
    fi
  fi
done
[[ -n "$base_python" ]] || {
  echo "compatible image Python is unavailable" >&2
  exit 69
}

verify_sidecar() {
  local input_file="$1"
  local sidecar="${input_file}.sha256"
  local expected=""
  local recorded=""
  local extra=""
  local actual=""
  [[ -f "$sidecar" ]] || { echo "SHA-256 sidecar is missing" >&2; exit 66; }
  read -r expected recorded extra < "$sidecar"
  [[ "$expected" =~ ^[0-9a-f]{64}$ && "$recorded" == "$(basename -- "$input_file")" && -z "$extra" ]] || {
    echo "SHA-256 sidecar is invalid" >&2
    exit 65
  }
  actual="$(sha256sum -- "$input_file")"
  actual="${actual%% *}"
  [[ "$actual" == "$expected" ]] || { echo "recovery kit SHA-256 mismatch" >&2; exit 65; }
}

verify_sidecar "$recovery_archive"
verify_sidecar "$server_bundle"

runtime_root="${repository_root}/artifacts/stage2/runtime"
logs_root="${repository_root}/logs"
venv_root="${repository_root}/.venv-stage2"
runtime_python="${venv_root}/bin/python"
requirements_lock="${repository_root}/deploy/stage2/requirements-server.lock"
runtime_marker="${venv_root}/.tarca-requirements-sha256"
expected_runtime_marker="$(sha256sum -- "$requirements_lock")"
expected_runtime_marker="${expected_runtime_marker%% *}"

current_stage="prepare-python"
if [[ ! -x "$runtime_python" ]]; then
  "$base_python" -m venv --system-site-packages --without-pip "$venv_root"
fi
installed_runtime_marker=""
if [[ -f "$runtime_marker" ]]; then
  installed_runtime_marker="$(<"$runtime_marker")"
fi
current_stage="install-dependencies"
if [[ "$installed_runtime_marker" != "$expected_runtime_marker" ]]; then
  "$runtime_python" -m pip install \
    --no-cache-dir \
    --no-index \
    --find-links="${repository_root}/deploy/stage2/wheelhouse" \
    --require-hashes \
    -r "$requirements_lock"
  printf '%s\n' "$expected_runtime_marker" > "$runtime_marker"
fi

export PATH="${venv_root}/bin:${PATH}"
export PYTHONPATH="${repository_root}/deploy/stage2/py310:${repository_root}/src"
export TARCA_EXECUTION_KIND="stage2-v1"
export TARCA_RUNTIME_DATABASE="${runtime_root}/execution.sqlite3"
export TARCA_RUNTIME_STATIC_ROOT="${repository_root}/frontend/stage1b-monitor/dist"

current_stage="verify-runtime"
"$runtime_python" -c \
  "import sys, torch; assert sys.version_info[:2] == (3, 10); assert torch.__version__.split('+')[0] == '2.2.2'"
current_stage="import-sources"
"$runtime_python" "${repository_root}/scripts/import_stage2_source_capsule.py" \
  --config "${repository_root}/configs/stage2/stage2_v1.yaml" \
  --cache-root "${repository_root}/third_party/stage2" \
  --capsule "${repository_root}/artifacts/stage2/source-capsules/stage2-v1-official-sources.tar.gz" \
  --receipt "${repository_root}/artifacts/stage2/source-capsules/stage2-v1-official-sources.tar.gz.receipt.json"

current_stage="restore"
"$runtime_python" "${repository_root}/scripts/run_stage2_v1.py" restore-input \
  --repository-root "$repository_root" \
  --config "${repository_root}/configs/stage2/stage2_v1.yaml" \
  --artifact-root "${repository_root}/artifacts/stage2" \
  --recovery-archive "$recovery_archive" \
  --server-bundle "$server_bundle"

current_stage="preflight"
bash "${repository_root}/deploy/stage2/bootstrap.sh" \
  --mode preflight \
  --remaining-rental-hours "$remaining_hours"

current_stage="repair"
"$runtime_python" "${repository_root}/scripts/run_stage2_v1.py" repair \
  --repository-root "$repository_root" \
  --config "${repository_root}/configs/stage2/stage2_v1.yaml" \
  --artifact-root "${repository_root}/artifacts/stage2" \
  --acknowledgement I_ACKNOWLEDGE_STAGE2_DEVICE_MISMATCH_RECOVERY_V1

current_stage="start-monitor"
mkdir -p "$runtime_root" "$logs_root"
monitor_pid_file="${runtime_root}/monitor.pid"
monitor_running="false"
if [[ -f "$monitor_pid_file" ]]; then
  monitor_pid="$(<"$monitor_pid_file")"
  if [[ "$monitor_pid" =~ ^[0-9]+$ ]] && kill -0 "$monitor_pid" 2>/dev/null; then
    monitor_running="true"
  fi
fi
if [[ "$monitor_running" != "true" ]]; then
  nohup setsid env \
    PATH="$PATH" \
    PYTHONPATH="$PYTHONPATH" \
    TARCA_EXECUTION_KIND="$TARCA_EXECUTION_KIND" \
    TARCA_RUNTIME_DATABASE="$TARCA_RUNTIME_DATABASE" \
    TARCA_RUNTIME_STATIC_ROOT="$TARCA_RUNTIME_STATIC_ROOT" \
    "$runtime_python" -m uvicorn tarca.monitoring.server:create_app_from_environment \
      --factory --host 127.0.0.1 --port 8765 \
      > "${logs_root}/stage2-monitor.log" 2>&1 < /dev/null &
  monitor_pid=$!
  printf '%s\n' "$monitor_pid" > "$monitor_pid_file"
fi

current_stage="verify-monitor"
monitor_ready="false"
for _ in $(seq 1 30); do
  if curl --fail --silent --show-error --max-time 1 \
    http://127.0.0.1:8765/api/v1/run >/dev/null; then
    monitor_ready="true"
    break
  fi
  sleep 1
done
[[ "$monitor_ready" == "true" ]] || {
  echo "read-only monitor did not become ready" >&2
  exit 70
}

printf '%s\n' "RECOVERY_READY_FOR_USER_RESUME"
printf '%s\n' "Read-only monitor: http://127.0.0.1:8765/"
printf '%s\n' "Direct container runtime prepared; resume remains a separate authorized action."
