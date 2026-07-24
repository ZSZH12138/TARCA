"""Command-line entry point for TARCA Stage 0 diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.stage0.diagnostics import (  # noqa: E402
    render_markdown,
    report_to_json,
    run_diagnostics,
)
from tarca.stage0.models import DoctorReport  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TARCA Stage 0 diagnostics.")
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--markdown", dest="markdown_path", type=Path)
    return parser.parse_args(argv)


def write_text_atomic(path: Path, content: str) -> None:
    """Atomically replace a UTF-8 text artifact in its destination directory."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".tarca-doctor-",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Callable[[Path], DoctorReport] = run_diagnostics,
    project_root: Path = PROJECT_ROOT,
) -> int:
    args = parse_args(argv)
    report = runner(project_root)
    markdown = render_markdown(report)
    sys.stdout.write(markdown)
    if args.json_path is not None:
        write_text_atomic(args.json_path, report_to_json(report))
    if args.markdown_path is not None:
        write_text_atomic(args.markdown_path, markdown)
    return 1 if report.has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
