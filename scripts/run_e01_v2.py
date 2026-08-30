from __future__ import annotations

import argparse
import json
from pathlib import Path

from tarca.e01.v2_runtime import dispatch_e01_v2_runtime_command


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operate the TARCA E01-v2 formal runtime")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/e01/e01_v2.yaml"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/e01-v2"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("dry-run")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--remaining-rental-hours", type=float, required=True)
    status = commands.add_parser("status")
    status.add_argument("--empty-ok", action="store_true")
    for name in ("launch", "resume"):
        command = commands.add_parser(name)
        command.add_argument("--acknowledgement", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    keyword: dict[str, object] = {}
    for name in ("remaining_rental_hours", "acknowledgement", "empty_ok"):
        if hasattr(arguments, name):
            keyword[name] = getattr(arguments, name)
    result = dispatch_e01_v2_runtime_command(
        arguments.command,
        arguments.repository_root,
        arguments.config,
        arguments.artifact_root,
        **keyword,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
