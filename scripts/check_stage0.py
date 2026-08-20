from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tarca.stage0.checks import complete_stage0, freeze_stage0, verify_stage0  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze or verify the TARCA Stage 0 contract.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--freeze",
        action="store_true",
        help="Freeze the research contract; the human Gate 0 decision remains a separate input.",
    )
    action.add_argument(
        "--complete",
        action="store_true",
        help="Publish the Stage 0 completion receipt after core verification succeeds.",
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON document.")
    parser.add_argument(
        "--skip-doctor",
        action="store_true",
        help="Skip numeric environment smoke; contract checks still run.",
    )
    parser.add_argument(
        "--allow-frozen-overwrite",
        action="store_true",
        help="Allow replacing frozen Stage 0 artifacts after explicit user authorization.",
    )
    parser.add_argument(
        "--authorization-reason",
        help="Audit reason required together with --allow-frozen-overwrite.",
    )
    args = parser.parse_args()
    try:
        if args.freeze:
            freeze_stage0(
                REPO_ROOT,
                allow_frozen_overwrite=args.allow_frozen_overwrite,
                authorization_reason=args.authorization_reason,
            )
            result = {
                "status": "FROZEN_PENDING_GATE_OR_COMPLETION",
                "next_action": ("Obtain the human GATE_0_NOVELTY decision, then run --complete."),
            }
        elif args.complete:
            complete_stage0(REPO_ROOT, run_doctor_check=not args.skip_doctor)
            result = verify_stage0(REPO_ROOT, run_doctor_check=not args.skip_doctor)
        else:
            result = verify_stage0(REPO_ROOT, run_doctor_check=not args.skip_doctor)
    except Exception as exc:
        result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        exit_code = 1
    else:
        exit_code = 0
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Stage 0 check: {result['status']}")
        if "error" in result:
            print(f"  error: {result['error']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
