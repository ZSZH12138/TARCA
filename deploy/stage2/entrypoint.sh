#!/usr/bin/env bash
set -euo pipefail
umask 027

case "${1:-monitor}" in
  monitor)
    exec python -m uvicorn tarca.monitoring.server:create_app_from_environment \
      --factory --host 0.0.0.0 --port 8765
    ;;
  bootstrap)
    shift
    exec bash /opt/tarca/deploy/stage2/bootstrap.sh "$@"
    ;;
  stage2|e02)
    exec bash /opt/tarca/deploy/stage2/supervisor.sh "$@"
    ;;
  *)
    echo "unsupported entrypoint mode" >&2
    exit 64
    ;;
esac
