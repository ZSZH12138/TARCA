from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from tarca.contracts import canonical_json_bytes, sha256_file
from tarca.e02.server_handoff import (
    E02ServerHandoffError,
    load_e02_server_handoff,
    restore_stage2_complete_archive,
    verify_server_bundle_checkout,
)

ROOT = Path(__file__).resolve().parents[2]


def _archive(tmp_path: Path, *, unsafe: bool = False) -> Path:
    archive = tmp_path / "stage2-complete.tar.gz"
    members = {
        "artifacts/stage2/frozen/v1/stage2_manifest.json": (
            ROOT / "artifacts/stage2/frozen/v1/stage2_manifest.json"
        ).read_bytes(),
        "artifacts/stage2/frozen/v1/stage2_freeze_receipt.json": (
            ROOT / "artifacts/stage2/frozen/v1/stage2_freeze_receipt.json"
        ).read_bytes(),
        "artifacts/stage2/runtime/execution.sqlite3": b"synthetic-complete-ledger",
        "artifacts/stage2/runtime/store/example.bin": b"fixed-artifact",
        "logs/stage2-resume.log": b"complete\n",
    }
    with tarfile.open(archive, "w:gz") as handle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
        if unsafe:
            escaped = tarfile.TarInfo("../escaped.txt")
            escaped.size = 6
            handle.addfile(escaped, io.BytesIO(b"escape"))
    return archive


def _handoff(tmp_path: Path, archive: Path) -> Path:
    receipt = json.loads(
        (ROOT / "artifacts/stage2/frozen/v1/stage2_freeze_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "schema_version": "tarca-e02-server-handoff-v1",
        "complete_archive_filename": archive.name,
        "complete_archive_sha256": sha256_file(archive),
        "stage2_freeze_receipt_sha256": receipt["receipt_sha256"],
        "stage2_freeze_receipt_file_sha256": sha256_file(
            ROOT / "artifacts/stage2/frozen/v1/stage2_freeze_receipt.json"
        ),
        "stage2_manifest_file_sha256": sha256_file(
            ROOT / "artifacts/stage2/frozen/v1/stage2_manifest.json"
        ),
        "e02_scientific_config_sha256": (
            "9027fcb9d40e89e66cd047c247b5dd5fc10a548916e9b97ffad45ddf262b310c"
        ),
        "required_members": [
            "artifacts/stage2/frozen/v1/stage2_manifest.json",
            "artifacts/stage2/frozen/v1/stage2_freeze_receipt.json",
            "artifacts/stage2/runtime/execution.sqlite3",
        ],
    }
    path = tmp_path / "handoff.json"
    path.write_bytes(canonical_json_bytes(payload) + b"\n")
    return path


def _repository(tmp_path: Path) -> Path:
    destination = tmp_path / "repository"
    config = destination / "configs/e02/e02_v1.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes((ROOT / "configs/e02/e02_v1.yaml").read_bytes())
    return destination


def test_repository_handoff_binds_the_authoritative_complete_archive() -> None:
    handoff = load_e02_server_handoff(ROOT / "configs/e02/e02_server_handoff_v1.json")

    assert handoff.complete_archive_sha256 == (
        "7d77ca6bd7d09b3fd9abd7814ef169f11814e774ec479a04d6f1a1c751fe966a"
    )
    assert handoff.stage2_freeze_receipt_sha256 == (
        "37a3a5c45a8bf4b4a703f40b7ab2e82b8d8b9bf0554a6b13f470ee7e5017e166"
    )


def test_complete_archive_restore_is_idempotent_and_seals_exact_identity(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    handoff = _handoff(tmp_path, archive)
    destination = _repository(tmp_path)

    first = restore_stage2_complete_archive(destination, archive, handoff)
    second = restore_stage2_complete_archive(destination, archive, handoff)

    assert first == second
    assert first["status"] == "RESTORED"
    assert first["complete_archive_sha256"] == sha256_file(archive)
    assert first["formal_tasks_executed"] == 0
    assert (destination / "artifacts/stage2/runtime/execution.sqlite3").read_bytes() == (
        b"synthetic-complete-ledger"
    )
    assert not (destination / "logs/stage2-resume.log").exists()
    assert (destination / "artifacts/e02/runtime/stage2_restore_receipt.json").is_file()


def test_complete_archive_restore_rejects_unsafe_member_before_writing(tmp_path: Path) -> None:
    archive = _archive(tmp_path, unsafe=True)
    handoff = _handoff(tmp_path, archive)
    destination = _repository(tmp_path)

    with pytest.raises(E02ServerHandoffError, match="unsafe"):
        restore_stage2_complete_archive(destination, archive, handoff)

    assert not (destination / "artifacts").exists()
    assert not (tmp_path / "escaped.txt").exists()


def test_complete_archive_restore_refuses_to_overwrite_drifted_file(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    handoff = _handoff(tmp_path, archive)
    destination = _repository(tmp_path)
    drifted = destination / "artifacts/stage2/runtime/store/example.bin"
    drifted.parent.mkdir(parents=True)
    drifted.write_bytes(b"user-state")

    with pytest.raises(E02ServerHandoffError, match="overwrite"):
        restore_stage2_complete_archive(destination, archive, handoff)

    assert drifted.read_bytes() == b"user-state"


def test_complete_archive_restore_rejects_hash_mismatch_before_writing(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    handoff_path = _handoff(tmp_path, archive)
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff["complete_archive_sha256"] = "0" * 64
    handoff_path.write_bytes(canonical_json_bytes(handoff) + b"\n")
    destination = _repository(tmp_path)

    with pytest.raises(E02ServerHandoffError, match="SHA-256"):
        restore_stage2_complete_archive(destination, archive, handoff_path)

    assert not (destination / "artifacts").exists()


def test_server_bundle_checkout_verification_binds_every_manifest_file(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    source = repository / "src/example.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"print('fixed')\n")
    sums = {"src/example.py": hashlib.sha256(source.read_bytes()).hexdigest()}
    bundle = tmp_path / "server.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        payload = canonical_json_bytes(
            {"schema_version": "tarca-stage2-bundle-v1", "files": sums}
        ) + b"\n"
        info = tarfile.TarInfo("SHA256SUMS.json")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))

    receipt = verify_server_bundle_checkout(repository, bundle)

    assert receipt["status"] == "VERIFIED"
    assert receipt["server_bundle_sha256"] == sha256_file(bundle)
    assert receipt["verified_file_count"] == 1
    assert (repository / "artifacts/e02/runtime/server_bundle_verification_receipt.json").is_file()

    source.write_bytes(b"print('drift')\n")
    with pytest.raises(E02ServerHandoffError, match="checkout file"):
        verify_server_bundle_checkout(repository, bundle)
