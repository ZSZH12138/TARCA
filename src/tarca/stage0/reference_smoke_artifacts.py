from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from tarca.stage0.reference_smoke_policy import MAX_LOG_BYTES, SmokeResult

_SECRET_KEY_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|token|password|secret|credential|authorization)"
)
_KEY_VALUE_SECRET_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|token|password|secret|credential|authorization)"
    r"[A-Za-z0-9_.-]*\s*[:=]\s*)([^\s]+)"
)


def redact_text(value: str) -> str:
    redacted = value
    for key, secret in os.environ.items():
        if _SECRET_KEY_PATTERN.search(key) is not None and len(secret) >= 4:
            redacted = redacted.replace(secret, "[REDACTED]")
    return _KEY_VALUE_SECRET_PATTERN.sub(r"\1[REDACTED]", redacted)


def truncate_log(value: str) -> str:
    redacted = redact_text(value)
    encoded = redacted.encode("utf-8")
    if len(encoded) <= MAX_LOG_BYTES:
        return redacted
    marker = b"\n[TRUNCATED]\n"
    prefix = encoded[: MAX_LOG_BYTES - len(marker)]
    return prefix.decode("utf-8", errors="ignore") + marker.decode("ascii")


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _status_payload(result: SmokeResult) -> dict[str, object]:
    return {
        "name": result.name,
        "command": [list(command) for command in result.command],
        "commit": result.commit,
        "exit_code": result.exit_code,
        "duration_seconds": float(result.duration_seconds),
        "used_gpu": result.used_gpu,
        "status": result.status.value,
        "expected_commit": result.expected_commit,
        "observed_commit": result.observed_commit,
        "phase": result.phase,
        "reason": redact_text(result.reason),
        "policy_error": result.policy_error,
        "checks": [
            {
                "name": check.name,
                "status": check.status,
                "detail": redact_text(check.detail),
            }
            for check in result.checks
        ],
        "generated_at": result.generated_at,
    }


def write_artifacts(directory: Path, result: SmokeResult) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    commands = "\n".join(subprocess.list2cmdline(command) for command in result.command)
    environment = json.dumps(
        dict(result.environment),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    payload = _status_payload(result)
    summary = "\n".join(
        (
            f"# {result.name} reference smoke",
            "",
            f"- Status: `{result.status.value}`",
            f"- Phase: `{result.phase}`",
            f"- Expected commit: `{result.expected_commit}`",
            f"- Observed commit: `{result.observed_commit or 'NOT_OBSERVED'}`",
            "- Used GPU: `false`",
            f"- Exit code: `{result.exit_code}`",
            f"- Duration seconds: `{result.duration_seconds:.3f}`",
            f"- Reason: {redact_text(result.reason)}",
            "",
            "This bounded Stage 0 check is not a paper-result reproduction.",
            "",
        )
    )
    files = {
        "commit.txt": f"{result.commit}\n",
        "environment.txt": f"{environment}\n",
        "command.txt": f"{commands}\n" if commands else "",
        "stdout.log": truncate_log(result.stdout),
        "stderr.log": truncate_log(result.stderr),
        "result_summary.md": summary,
        "status.json": json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
    }
    for name, value in files.items():
        _atomic_write_text(directory / name, value)
