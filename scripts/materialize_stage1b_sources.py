from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from tarca.stage1b.config import load_world_suite
from tarca.stage1b.sources import (
    SourceMaterializationReceipt,
    SubprocessGitRunner,
    materialize_source,
    source_acquisition_mode_from_environment,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize hash-pinned Stage1B official sources without executing them."
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
        "--source-id",
        action="append",
        default=[],
        help="Materialize only this registered source ID; repeat for multiple sources.",
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


def _receipt_payload(receipt: SourceMaterializationReceipt) -> dict[str, Any]:
    payload = _json_value(asdict(receipt))
    if not isinstance(payload, dict):
        raise TypeError("receipt serialization must produce an object")
    return payload


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    suite = load_world_suite(arguments.config.resolve())
    requested = tuple(arguments.source_id)
    registered = {source.source_id: source for source in suite.sources}
    unknown = tuple(sorted(set(requested) - set(registered)))
    if unknown:
        _parser().error(f"unknown source IDs: {', '.join(unknown)}")
    selected = suite.sources if not requested else tuple(registered[item] for item in requested)
    runner = SubprocessGitRunner.discover()
    receipts = tuple(
        materialize_source(
            source,
            arguments.cache_root.resolve(),
            runner,
            mode=source_acquisition_mode_from_environment(),
        )
        for source in selected
    )
    print(
        json.dumps(
            {"schema_version": "2.0.0", "receipts": [_receipt_payload(item) for item in receipts]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
