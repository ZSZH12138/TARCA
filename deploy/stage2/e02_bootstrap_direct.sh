#!/usr/bin/env bash
set -euo pipefail
umask 027

repository_root=""
stage2_archive=""
server_bundle=""
remaining_hours=""
use_current_python="false"
while (($#)); do
  case "$1" in
    --repository-root) repository_root="${2:?missing repository root}"; shift 2 ;;
    --stage2-archive) stage2_archive="${2:?missing Stage 2 archive}"; shift 2 ;;
    --server-bundle) server_bundle="${2:?missing server bundle}"; shift 2 ;;
    --remaining-rental-hours) remaining_hours="${2:?missing hours}"; shift 2 ;;
    --use-current-python) use_current_python="true"; shift ;;
    *) printf '%s\n' "unknown E02 direct-bootstrap argument: $1" >&2; exit 64 ;;
  esac
done

[[ -n "$repository_root" && -n "$stage2_archive" && -n "$server_bundle" ]] || {
  printf '%s\n' "repository root, Stage 2 archive, and server bundle are required" >&2
  exit 64
}
[[ "$remaining_hours" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  printf '%s\n' "valid remaining rental hours are required" >&2
  exit 64
}

repository_root="$(realpath -e -- "$repository_root")"
stage2_archive="$(realpath -e -- "$stage2_archive")"
server_bundle="$(realpath -e -- "$server_bundle")"
[[ -f "$stage2_archive" && -f "$server_bundle" ]] || {
  printf '%s\n' "E02 handoff files must be regular files" >&2
  exit 66
}

verify_sidecar() {
  local target="$1"
  local sidecar="${target}.sha256"
  [[ -f "$sidecar" ]] || { printf '%s\n' "SHA-256 sidecar is missing" >&2; exit 66; }
  (cd "$(dirname -- "$target")" && sha256sum --check --status "$(basename -- "$sidecar")")
}

verify_sidecar "$stage2_archive"
verify_sidecar "$server_bundle"

runtime_python="$(command -v python)"
[[ -n "$runtime_python" ]] || { printf '%s\n' "Python is unavailable" >&2; exit 69; }
if [[ "$use_current_python" != "true" ]]; then
  venv_root="${repository_root}/artifacts/e02/runtime/server-venv"
  "$runtime_python" -m venv --system-site-packages "$venv_root"
  runtime_python="${venv_root}/bin/python"
  "$runtime_python" -m pip install \
    --no-index \
    --find-links="${repository_root}/deploy/stage2/wheelhouse" \
    --require-hashes \
    -r "${repository_root}/deploy/stage2/requirements-server.lock"
fi

export PYTHONPATH="${repository_root}/deploy/stage2/py310:${repository_root}/src"
cd "$repository_root"

"$runtime_python" -m tarca.e02.server_handoff verify-bundle \
  --repository-root "$repository_root" \
  --server-bundle "$server_bundle"
"$runtime_python" "${repository_root}/scripts/import_stage2_source_capsule.py" \
  --config "${repository_root}/configs/stage2/stage2_v1.yaml" \
  --cache-root "${repository_root}/third_party/stage2" \
  --capsule "${repository_root}/artifacts/stage2/source-capsules/stage2-v1-official-sources.tar.gz" \
  --receipt "${repository_root}/artifacts/stage2/source-capsules/stage2-v1-official-sources.tar.gz.receipt.json"
"$runtime_python" -m tarca.e02.server_handoff restore \
  --repository-root "$repository_root" \
  --stage2-archive "$stage2_archive" \
  --handoff "${repository_root}/configs/e02/e02_server_handoff_v1.json"

"$runtime_python" "${repository_root}/scripts/run_e02_v1.py" prepare \
  --repository-root "$repository_root" \
  --config "${repository_root}/configs/e02/e02_v1.yaml" \
  --artifact-root "${repository_root}/artifacts/e02"
"$runtime_python" "${repository_root}/scripts/run_e02_v1.py" dry-run \
  --repository-root "$repository_root" \
  --config "${repository_root}/configs/e02/e02_v1.yaml" \
  --artifact-root "${repository_root}/artifacts/e02"
"$runtime_python" -m tarca.e02.server_preflight \
  --remaining-rental-hours "$remaining_hours" \
  --repository-root "$repository_root" \
  --e02-config "${repository_root}/configs/e02/e02_v1.yaml" \
  --stage2-config "${repository_root}/configs/stage2/stage2_v1.yaml" \
  --artifact-root "${repository_root}/artifacts/e02" \
  --handoff "${repository_root}/configs/e02/e02_server_handoff_v1.json"
"$runtime_python" "${repository_root}/scripts/run_e02_v1.py" preflight \
  --repository-root "$repository_root" \
  --config "${repository_root}/configs/e02/e02_v1.yaml" \
  --artifact-root "${repository_root}/artifacts/e02" \
  --evidence "${repository_root}/artifacts/e02/runtime/bootstrap_evidence.json"

for forbidden in \
  "${repository_root}/artifacts/e02/runtime/sealed_access_grant.json" \
  "${repository_root}/artifacts/e02/runtime/execution.sqlite3" \
  "${repository_root}/artifacts/e02/frozen/v1/e02_receipt.json"; do
  [[ ! -e "$forbidden" ]] || {
    printf '%s\n' "E02 bootstrap crossed the formal execution boundary" >&2
    exit 70
  }
done

printf '%s\n' "E02_READY_FOR_USER_LAUNCH"
printf '%s\n' "Formal data remains sealed; a separate explicit launch authorization is required."
