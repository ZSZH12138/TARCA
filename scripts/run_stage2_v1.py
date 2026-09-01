from __future__ import annotations

import argparse
import json
from pathlib import Path

from tarca.stage2.runtime import dispatch_stage2_runtime_command


def main() -> int:
    parser = argparse.ArgumentParser(description="TARCA Stage 2 v1 runtime")
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "dry-run",
            "preflight",
            "launch",
            "restore-input",
            "repair",
            "resume",
            "status",
            "freeze",
            "recover",
        ),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/stage2/stage2_v1.yaml"))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/stage2"))
    parser.add_argument("--acknowledgement", default="")
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--recovery-archive", type=Path)
    parser.add_argument("--server-bundle", type=Path)
    args = parser.parse_args()
    keyword = {}
    if args.command in {"launch", "repair", "resume"}:
        keyword["acknowledgement"] = args.acknowledgement
    if args.command == "preflight":
        keyword["evidence_path"] = args.evidence
    if args.command == "restore-input":
        if args.recovery_archive is None or args.server_bundle is None:
            parser.error("restore-input requires --recovery-archive and --server-bundle")
        keyword["recovery_archive"] = args.recovery_archive
        keyword["server_bundle"] = args.server_bundle
    result = dispatch_stage2_runtime_command(
        args.command, args.repository_root, args.config, args.artifact_root, **keyword
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
