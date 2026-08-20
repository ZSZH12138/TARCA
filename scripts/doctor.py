from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tarca.stage0.environment import run_doctor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run TARCA Stage 0 CPU/offline smoke checks.")
    parser.add_argument("--json", action="store_true", help="Emit one JSON document.")
    args = parser.parse_args()
    result = run_doctor(REPO_ROOT)
    payload = result.model_dump(mode="json", exclude_none=True)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Stage 0 doctor: {result.status}")
        for name, status in sorted(result.checks.model_dump(exclude_none=True).items()):
            print(f"  {name}: {status}")
        if result.error is not None:
            print(f"  error: {result.error}")
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
