from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from scripts.prepare_e01_v2_server_bundle import build_e01_v2_server_bundle
from tarca.contracts import canonical_json_hash
from tarca.e01.v2_carry_forward import CarryForwardVerificationError

ROOT = Path(__file__).resolve().parents[2]


def _archive_payloads(path: Path) -> tuple[dict[str, tarfile.TarInfo], dict[str, bytes]]:
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        payloads = {
            name: archive.extractfile(member).read()
            for name, member in members.items()
            if member.isfile()
        }
    return members, payloads


def test_v2_bundle_is_deterministic_sealed_and_minimal(tmp_path: Path) -> None:
    first_path = tmp_path / "first.tar.gz"
    second_path = tmp_path / "second.tar.gz"

    first = build_e01_v2_server_bundle(ROOT, first_path)
    second = build_e01_v2_server_bundle(ROOT, second_path)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["bundle_sha256"] == hashlib.sha256(first_path.read_bytes()).hexdigest()
    assert (
        Path(f"{first_path}.sha256").read_text(encoding="ascii").strip() == first["bundle_sha256"]
    )
    receipt_path = tmp_path / "first.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["receipt_sha256"] == canonical_json_hash(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )

    members, payloads = _archive_payloads(first_path)
    required = {
        "TARCA/configs/e01/e01_v2.yaml",
        "TARCA/scripts/run_e01_v2.py",
        "TARCA/deploy/e01/Dockerfile.v2",
        "TARCA/deploy/e01/server_bootstrap_v2.sh",
        "TARCA/deploy/e01/server_supervisor_v2.sh",
        "TARCA/artifacts/e01/history/e01_v1_history_record.json",
        "TARCA/artifacts/stage1b/active.json",
        "TARCA/docs/auth/TARCA_PROTOCOL_CHANGE_CONTROL_CCP_0004.md",
        "TARCA/docs/research/e01_execution_spec_v2.md",
        "TARCA/docs/research/e01_server_handoff_v2.md",
        "TARCA/SHA256SUMS.json",
    }
    assert required <= members.keys()
    assert not any("artifacts/e01/recovery" in name for name in members)
    assert not any("artifacts/e01/formal" in name for name in members)
    assert not any("artifacts/e01/carry-forward" in name for name in members)
    assert not any("source-capsules" in name for name in members)
    assert "TARCA/deploy/e01/entrypoint.sh" not in members
    assert "TARCA/deploy/e01/server_supervisor.sh" not in members
    assert not any(name.endswith(".tsbuildinfo") for name in members)
    assert not any("__pycache__" in name or "node_modules" in name for name in members)

    manifest = json.loads(payloads["TARCA/SHA256SUMS.json"])
    assert set(manifest) == set(payloads) - {"TARCA/SHA256SUMS.json"}
    assert all(
        hashlib.sha256(payloads[name]).hexdigest() == digest for name, digest in manifest.items()
    )
    shell_members = [member for name, member in members.items() if name.endswith(".sh")]
    assert shell_members and all(member.mode == 0o755 for member in shell_members)
    assert not any(b"-----BEGIN PRIVATE KEY-----" in payload for payload in payloads.values())
    assert not any(b"C:\\Users\\DELL" in payload for payload in payloads.values())


def test_v2_bundle_rejects_v1_history_record_drift(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    config = repository / "configs/e01/e01_v2.yaml"
    config.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "configs/e01/e01_v2.yaml", config)
    history = repository / "artifacts/e01/history/e01_v1_history_record.json"
    history.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "artifacts/e01/history/e01_v1_history_record.json", history)
    with history.open("ab") as stream:
        stream.write(b"\n")

    with pytest.raises(CarryForwardVerificationError, match="history record SHA-256"):
        build_e01_v2_server_bundle(repository, tmp_path / "drift.tar.gz")
