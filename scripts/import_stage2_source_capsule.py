from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tarca.stage1b.source_capsules import (  # noqa: E402
    SourceCapsuleVerificationError,
    source_capsule_receipt_path,
)
from tarca.stage1b.sources import SourceVerificationError  # noqa: E402
from tarca.stage2.config import load_stage2_config  # noqa: E402
from tarca.stage2.sources import import_stage2_source_capsule  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import an audited Stage 2 source capsule without network access."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/stage2/stage2_v1.yaml",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPOSITORY_ROOT / "third_party/stage2",
    )
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    capsule = arguments.capsule.resolve()
    receipt = (
        arguments.receipt.resolve()
        if arguments.receipt is not None
        else source_capsule_receipt_path(capsule)
    )
    try:
        materialized = import_stage2_source_capsule(
            load_stage2_config(arguments.config.resolve()),
            capsule,
            receipt,
            arguments.cache_root.resolve(),
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
                        "source_id": item.source_id,
                        "commit": item.commit,
                        "tree_sha256": item.tree_sha256,
                    }
                    for item in materialized
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

