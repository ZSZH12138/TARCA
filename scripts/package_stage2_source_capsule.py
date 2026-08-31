from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tarca.stage1b.source_capsules import (  # noqa: E402
    source_capsule_receipt_path,
    source_capsule_receipt_payload,
)
from tarca.stage2.config import load_stage2_config  # noqa: E402
from tarca.stage2.sources import build_stage2_source_capsule  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the audited offline Stage 2 official-source capsule."
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts/stage2/source-capsules/stage2-v1-official-sources.tar.gz",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.resolve()
    receipt = build_stage2_source_capsule(
        load_stage2_config(arguments.config.resolve()),
        arguments.cache_root.resolve(),
        output,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "capsule": str(output),
                "receipt": str(source_capsule_receipt_path(output)),
                **source_capsule_receipt_payload(receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

