from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tarca.stage1b.sources import SourceAcquisitionMode  # noqa: E402
from tarca.stage2.config import load_stage2_config  # noqa: E402
from tarca.stage2.sources import materialize_stage2_sources  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize the four hash-pinned Stage 2 official sources."
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
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Refuse network acquisition and only verify an imported source cache.",
    )
    return parser


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    mode = (
        SourceAcquisitionMode.OFFLINE_CAPSULE
        if arguments.offline
        else SourceAcquisitionMode.ONLINE
    )
    materialized = materialize_stage2_sources(
        load_stage2_config(arguments.config.resolve()),
        arguments.cache_root.resolve(),
        source_ids=tuple(arguments.source_id),
        mode=mode,
    )
    print(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "receipts": [
                    _json_value(asdict(receipt)) for receipt in materialized.receipts
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

