from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.stage0.reference_smoke import (  # noqa: E402
    MAX_TIMEOUT_SECONDS,
    SmokeStatus,
    run_reference_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one fixed, CPU-only TARCA Stage 0 reference smoke."
    )
    parser.add_argument("name", choices=("plot", "diroca"))
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help=f"Per-command timeout in seconds (1-{MAX_TIMEOUT_SECONDS}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = run_reference_smoke(
            arguments.name,
            cache_root=Path(".cache/third_party"),
            artifact_root=Path("artifacts/stage0/reference_smoke"),
            timeout_seconds=arguments.timeout,
        )
    except ValueError as error:
        print(json.dumps({"status": "INPUT_ERROR", "reason": str(error)}))
        return 2
    print(
        json.dumps(
            {
                "name": result.name,
                "status": result.status.value,
                "phase": result.phase,
                "artifact_directory": (f"artifacts/stage0/reference_smoke/{result.name}"),
            },
            ensure_ascii=False,
        )
    )
    if result.policy_error:
        return 2
    if result.status is SmokeStatus.FAILED:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
