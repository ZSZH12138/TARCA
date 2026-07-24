from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from third_party_manifest import record_commits  # noqa: E402

import tarca.stage0.sources as sources_module  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"
MANIFEST_PATH = PROJECT_ROOT / "third_party_manifest" / "sources.yaml"


def test_cli_invalid_schema_returns_two_without_traceback(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.yaml"
    output = tmp_path / "result.json"
    manifest.write_text("- name: incomplete\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "third_party_manifest" / "record_commits.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "invalid manifest" in result.stderr.lower()
    assert "traceback" not in result.stderr.lower()
    assert not output.exists()


def test_cli_writes_every_resolution_and_preserves_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "sources.yaml"
    output = tmp_path / "result.json"
    raw_entries = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))[:2]
    isolated_entries = raw_entries
    manifest.write_text(
        yaml.safe_dump(isolated_entries, sort_keys=False),
        encoding="utf-8",
    )
    original_bytes = manifest.read_bytes()

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if isolated_entries[0]["repository_url"] in args:
            return subprocess.CompletedProcess(
                args,
                128,
                "",
                "fatal: unable to access remote",
            )
        return subprocess.CompletedProcess(
            args,
            0,
            f"{SHA}\trefs/heads/{isolated_entries[1]['default_branch']}\n",
            "",
        )

    monkeypatch.setattr(sources_module, "run", fake_run)
    monkeypatch.setattr(sources_module, "PROJECT_ROOT", tmp_path)

    return_code = record_commits.main(["--manifest", str(manifest), "--output", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert return_code == 0
    assert [result["status"] for result in payload["results"]] == [
        "NETWORK_ERROR",
        "VERIFIED",
    ]
    assert len(payload["results"]) == len(isolated_entries)
    assert manifest.read_bytes() == original_bytes
    assert set(tmp_path.iterdir()) == {manifest, output}
    assert payload["generated_at"].endswith("+00:00")
