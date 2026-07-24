from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.stage0.sources import load_sources, resolve_source  # noqa: E402

DEFAULT_MANIFEST = Path("third_party_manifest/sources.yaml")
DEFAULT_OUTPUT = Path("artifacts/stage0/third_party_commits.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record verified third-party commits.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        sources = load_sources(args.manifest)
    except (OSError, UnicodeError, ValueError, ValidationError, yaml.YAMLError) as error:
        summary = str(error).splitlines()[0]
        print(f"Invalid manifest {args.manifest}: {summary}", file=sys.stderr)
        return 2

    results = [asdict(resolve_source(entry)) for entry in sources]
    payload = {
        "manifest": str(args.manifest).replace("\\", "/"),
        "generated_at": datetime.now(UTC).isoformat(),
        "results": results,
    }
    atomic_write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
