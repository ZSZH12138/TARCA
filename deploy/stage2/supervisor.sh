#!/usr/bin/env bash
set -euo pipefail
umask 027

kind="${1:?stage2 or e02 required}"
shift
case "$kind" in
  stage2)
    exec python /opt/tarca/scripts/run_stage2_v1.py "$@"
    ;;
  e02)
    export TARCA_EXECUTION_KIND=e02-v1
    exec python /opt/tarca/scripts/run_e02_v1.py "$@"
    ;;
  *)
    echo "unsupported supervisor kind" >&2
    exit 64
    ;;
esac
