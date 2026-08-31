from __future__ import annotations

import argparse
import json
from pathlib import Path

from tarca.e02.runtime import dispatch_e02_runtime_command


def main() -> int:
    parser = argparse.ArgumentParser(description="TARCA E02 v1 formal runtime")
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "dry-run",
            "preflight",
            "launch",
            "resume",
            "status",
            "finalize",
            "recover",
        ),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/e02/e02_v1.yaml"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/e02"))
    parser.add_argument("--acknowledgement", default="")
    args = parser.parse_args()
    keyword = (
        {"acknowledgement": args.acknowledgement} if args.command in {"launch", "resume"} else {}
    )
    result = dispatch_e02_runtime_command(
        args.command, args.repository_root, args.config, args.artifact_root, **keyword
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
