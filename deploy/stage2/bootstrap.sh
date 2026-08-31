#!/usr/bin/env bash
set -euo pipefail
umask 027

mode=""
remaining_hours=""
while (($#)); do
  case "$1" in
    --mode) mode="${2:?missing mode}"; shift 2 ;;
    --remaining-rental-hours) remaining_hours="${2:?missing hours}"; shift 2 ;;
    *) echo "unknown bootstrap argument: $1" >&2; exit 64 ;;
  esac
done
[[ "$mode" == "preflight" ]] || { echo "--mode preflight is required" >&2; exit 64; }
[[ "$remaining_hours" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "valid remaining rental hours required" >&2; exit 64; }

cd /opt/tarca
mkdir -p artifacts/stage2/runtime
python scripts/run_stage2_v1.py prepare \
  --repository-root /opt/tarca --config configs/stage2/stage2_v1.yaml \
  --artifact-root artifacts/stage2
python -m tarca.stage2.server_preflight \
  --remaining-rental-hours "$remaining_hours" \
  --repository-root /opt/tarca \
  --config configs/stage2/stage2_v1.yaml \
  --artifact-root artifacts/stage2
python scripts/run_stage2_v1.py preflight \
  --repository-root /opt/tarca --config configs/stage2/stage2_v1.yaml \
  --artifact-root artifacts/stage2 \
  --evidence artifacts/stage2/runtime/bootstrap_evidence.json
echo "PREFLIGHT_PASS: no training or formal task was started"
