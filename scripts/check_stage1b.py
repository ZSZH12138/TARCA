from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tarca.stage1b.freeze import verify_frozen_suite  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the TARCA Stage1B frozen suite.")
    parser.add_argument("--version")
    parser.add_argument("--allow-unfrozen", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/stage1b",
    )
    args = parser.parse_args()
    try:
        result = verify_frozen_suite(
            args.artifact_root,
            version=args.version,
            allow_unfrozen=args.allow_unfrozen,
        )
    except Exception as exc:
        result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        exit_code = 1
    else:
        exit_code = 0
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Stage1B check: {result['status']}")
        if "error" in result:
            print(f"  error: {result['error']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
