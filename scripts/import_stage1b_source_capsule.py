from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tarca.stage1b.config import load_world_suite  # noqa: E402
from tarca.stage1b.source_capsules import (  # noqa: E402
    SourceCapsuleVerificationError,
    import_source_capsule,
    source_capsule_receipt_path,
)
from tarca.stage1b.sources import SourceVerificationError, SubprocessGitRunner  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import a locally audited Stage1B source capsule without contacting GitHub."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPOSITORY_ROOT / "third_party/stage1b",
    )
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="Defaults to <capsule>.receipt.json beside the uploaded capsule.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    capsule = arguments.capsule.resolve()
    receipt_path = (
        arguments.receipt.resolve()
        if arguments.receipt is not None
        else source_capsule_receipt_path(capsule)
    )
    try:
        receipts = import_source_capsule(
            load_world_suite(arguments.config.resolve()).sources,
            capsule,
            receipt_path,
            arguments.cache_root.resolve(),
            SubprocessGitRunner.discover(),
        )
    except (OSError, SourceCapsuleVerificationError, SourceVerificationError, ValueError) as error:
        print(
            json.dumps(
                {"status": "FAIL", "error": f"{type(error).__name__}: {error}"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "sources": [
                    {
                        "source_id": receipt.source_id,
                        "commit": receipt.commit,
                        "tree_sha256": receipt.tree_sha256,
                    }
                    for receipt in receipts
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
