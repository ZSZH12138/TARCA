#!/usr/bin/env bash
set -euo pipefail
umask 027

recovery_archive=""
server_bundle=""
remaining_hours=""

while (($#)); do
  case "$1" in
    --recovery-archive) recovery_archive="${2:?missing recovery archive}"; shift 2 ;;
    --server-bundle) server_bundle="${2:?missing server bundle}"; shift 2 ;;
    --remaining-rental-hours) remaining_hours="${2:?missing remaining hours}"; shift 2 ;;
    *) echo "unsupported recovery bootstrap argument" >&2; exit 64 ;;
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

recovery_archive="$(realpath -e -- "$recovery_archive")"
server_bundle="$(realpath -e -- "$server_bundle")"
[[ -f "$recovery_archive" && -f "$server_bundle" ]] || {
  echo "recovery kit input is not a regular file" >&2
  exit 66
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

script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd -- "$script_root"
compose=(docker compose -f deploy/stage2/compose.yaml)
recovery_container_path="/recovery/$(basename -- "$recovery_archive")"

"${compose[@]}" build tarca-stage2
mkdir -p artifacts/stage2/runtime
"${compose[@]}" run --rm --user 0:0 \
  -v "${recovery_archive}:${recovery_container_path}:ro" \
  -v "${server_bundle}:/recovery/server.tar.gz:ro" \
  tarca-stage2 stage2 restore-input \
  --repository-root /opt/tarca \
  --config configs/stage2/stage2_v1.yaml \
  --artifact-root artifacts/stage2 \
  --recovery-archive "${recovery_container_path}" \
  --server-bundle /recovery/server.tar.gz
"${compose[@]}" run --rm --user 0:0 --entrypoint sh tarca-stage2 \
  -c 'chown -R tarca:tarca /opt/tarca/artifacts'
"${compose[@]}" run --rm tarca-stage2 \
  bootstrap --mode preflight --remaining-rental-hours "$remaining_hours"
"${compose[@]}" run --rm tarca-stage2 stage2 repair \
  --repository-root /opt/tarca \
  --config configs/stage2/stage2_v1.yaml \
  --artifact-root artifacts/stage2 \
  --acknowledgement I_ACKNOWLEDGE_STAGE2_DEVICE_MISMATCH_RECOVERY_V1
"${compose[@]}" up -d tarca-stage2

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
  "${compose[@]}" logs --tail 50 tarca-stage2 >&2
  echo "read-only monitor did not become ready" >&2
  exit 70
}

resume_command="docker compose -f deploy/stage2/compose.yaml run -d --name tarca-stage2-recovery-resume tarca-stage2 stage2 resume --repository-root /opt/tarca --config configs/stage2/stage2_v1.yaml --artifact-root artifacts/stage2 --acknowledgement I_ACKNOWLEDGE_STAGE2_V1_TRAINING_RUN"
printf '%s\n' "RECOVERY_READY_FOR_USER_RESUME"
printf '%s\n' "Read-only monitor: http://127.0.0.1:8765/"
printf '%s\n' "Run the following command only after the user confirms resume:"
printf '%s\n' "${resume_command}"
