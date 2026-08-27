from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tarca.stage1b.config import load_world_suite  # noqa: E402
from tarca.stage1b.source_capsules import (  # noqa: E402
    build_source_capsule,
    source_capsule_receipt_path,
    source_capsule_receipt_payload,
)
from tarca.stage1b.sources import SubprocessGitRunner  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an auditable offline Stage1B official-source capsule on this local machine."
        )
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
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT
        / "artifacts/stage1b/source-capsules/stage1b-v2-official-sources.tar.gz",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    output = arguments.output.resolve()
    receipt = build_source_capsule(
        load_world_suite(arguments.config.resolve()).sources,
        arguments.cache_root.resolve(),
        output,
        SubprocessGitRunner.discover(),
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
