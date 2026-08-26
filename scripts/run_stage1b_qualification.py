from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tarca.stage1b.freeze import (  # noqa: E402
    OverrideAuthorization,
    freeze_suite,
)
from tarca.stage1b.runner import (  # noqa: E402
    run_hardware_probe,
    run_scheduled_qualification,
)


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--worlds",
        type=Path,
        default=REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml",
    )
    parser.add_argument(
        "--qualification",
        type=Path,
        default=REPOSITORY_ROOT / "configs/stage1b/qualification_v2.yaml",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/stage1b",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run independent TARCA Stage1B world qualification."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe", help="Measure and extrapolate the fixed workload.")
    _common_paths(probe)
    qualify = subparsers.add_parser("qualify", help="Run the approved qualification-only workload.")
    _common_paths(qualify)
    freeze = subparsers.add_parser("freeze", help="Freeze an automatically passing receipt.")
    freeze.add_argument(
        "--receipt",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/stage1b/qualification_v2_summary.json",
    )
    freeze.add_argument(
        "--artifact-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/stage1b",
    )
    freeze.add_argument("--series", default="v2", choices=("v2",))
    freeze.add_argument("--revision-id", default="v2-r1")
    freeze.add_argument("--authorize-override", action="store_true")
    freeze.add_argument("--prior-revision-id")
    freeze.add_argument("--authorization-reason")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "probe":
            result = run_hardware_probe(
                args.worlds,
                args.qualification,
                args.artifact_root / "runtime",
            )
        elif args.command == "qualify":
            official_worlds = (REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml").resolve()
            if args.worlds.resolve() != official_worlds:
                raise ValueError("scheduled qualification uses the official v2 world configuration")
            if (
                args.qualification.resolve()
                != (REPOSITORY_ROOT / "configs/stage1b/qualification_v2.yaml").resolve()
            ):
                raise ValueError(
                    "scheduled qualification uses the official v2 qualification configuration"
                )
            result = run_scheduled_qualification(
                REPOSITORY_ROOT,
                args.artifact_root,
            )
        else:
            receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
            authorization = None
            if args.authorize_override:
                if not args.prior_revision_id or not args.authorization_reason:
                    raise ValueError(
                        "authorized override requires prior revision and authorization reason"
                    )
                authorization = OverrideAuthorization(
                    authorized_by="user",
                    reason=args.authorization_reason,
                    prior_revision_id=args.prior_revision_id,
                )
            result = freeze_suite(
                receipt,
                args.artifact_root,
                series=args.series,
                revision_id=args.revision_id,
                authorization=authorization,
            )
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
