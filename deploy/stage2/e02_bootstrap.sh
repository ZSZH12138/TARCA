#!/usr/bin/env bash
set -euo pipefail
umask 027

stage2_archive=""
server_bundle=""
remaining_hours=""
while (($#)); do
  case "$1" in
    --stage2-archive) stage2_archive="${2:?missing Stage 2 archive}"; shift 2 ;;
    --server-bundle) server_bundle="${2:?missing server bundle}"; shift 2 ;;
    --remaining-rental-hours) remaining_hours="${2:?missing hours}"; shift 2 ;;
    *) printf '%s\n' "unknown E02 bootstrap argument: $1" >&2; exit 64 ;;
  esac
done

[[ -n "$stage2_archive" && -n "$server_bundle" ]] || {
  printf '%s\n' "Stage 2 archive and server bundle are required" >&2
  exit 64
}
[[ "$remaining_hours" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  printf '%s\n' "valid remaining rental hours are required" >&2
  exit 64
}
command -v docker >/dev/null 2>&1 || {
  printf '%s\n' "Docker is unavailable; use e02_bootstrap_direct.sh inside the target container" >&2
  exit 69
}

stage2_archive="$(realpath -e -- "$stage2_archive")"
server_bundle="$(realpath -e -- "$server_bundle")"

verify_sidecar() {
  local target="$1"
  local sidecar="${target}.sha256"
  [[ -f "$target" && -f "$sidecar" ]] || {
    printf '%s\n' "archive and SHA-256 sidecar are both required" >&2
    exit 66
  }
  (cd "$(dirname -- "$target")" && sha256sum --check --status "$(basename -- "$sidecar")")
}

verify_sidecar "$stage2_archive"
verify_sidecar "$server_bundle"

repository_root="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$repository_root"
docker compose -f deploy/stage2/compose.yaml build tarca-stage2

archive_name="$(basename -- "$stage2_archive")"
bundle_name="$(basename -- "$server_bundle")"
docker compose -f deploy/stage2/compose.yaml run --rm \
  -v "${stage2_archive}:/handoff/${archive_name}:ro" \
  -v "${stage2_archive}.sha256:/handoff/${archive_name}.sha256:ro" \
  -v "${server_bundle}:/handoff/${bundle_name}:ro" \
  -v "${server_bundle}.sha256:/handoff/${bundle_name}.sha256:ro" \
  tarca-stage2 e02-bootstrap \
  --repository-root /opt/tarca \
  --stage2-archive "/handoff/${archive_name}" \
  --server-bundle "/handoff/${bundle_name}" \
  --remaining-rental-hours "$remaining_hours" \
  --use-current-python
