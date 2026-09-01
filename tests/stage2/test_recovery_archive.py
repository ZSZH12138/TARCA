from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from tarca.contracts import canonical_json_bytes, canonical_json_hash, sha256_file
from tests.stage2.test_recovery import _recovery_fixture


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o640
    archive.addfile(info, io.BytesIO(payload))


def _synthetic_archive(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source_root, _, spec_path, _ = _recovery_fixture(tmp_path / "source")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    snapshot = source_root / "artifacts/stage2/runtime/execution.sqlite3"
    manifest_payload: dict[str, object] = {
        "schema_version": "tarca-stage2-server-recovery-archive-v1",
        "created_at_utc": "2026-08-31T10:21:55.536180+00:00",
        "run": {
            "run_id": spec["run_id"],
            "graph_id": spec["graph_id"],
            "status": "ACTIVE",
        },
        "planned_tasks": spec["planned_task_count"],
        "failed_neural_tasks": 6,
        "attempt_counts": {
            "COMPLETED": spec["completed_attempt_count"],
            "FAILED": 6,
        },
        "database_snapshot_relative_path": "transfer/stage2-snapshot.sqlite3",
        "database_snapshot_sha256": sha256_file(snapshot),
        "checkpoints": [
            {
                "relative_path": task["checkpoint_relative_path"],
                "sha256": task["checkpoint_sha256"],
                "status": "COMPLETE",
                "epoch": task["checkpoint_epoch"],
                "seed": task["seed"],
            }
            for task in spec["tasks"]
        ],
        "frozen_input_sha256": {},
    }
    manifest_payload["manifest_sha256"] = canonical_json_hash(manifest_payload)
    manifest_name = "transfer/stage2-recovery-manifest-20260831T102151Z.json"
    archive_path = tmp_path / "synthetic-recovery.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        _add_bytes(
            archive,
            manifest_name,
            canonical_json_bytes(manifest_payload) + b"\n",
        )
        _add_bytes(
            archive,
            "transfer/stage2-snapshot.sqlite3",
            snapshot.read_bytes(),
        )
        _add_bytes(
            archive,
            "artifacts/stage2/runtime/execution.sqlite3",
            b"stale-live-database",
        )
        _add_bytes(
            archive,
            "artifacts/stage2/runtime/store/marker.bin",
            b"restored-artifact",
        )
        _add_bytes(
            archive,
            "src/tarca/stage2/training.py",
            b"old-buggy-source",
        )
        for task in spec["tasks"]:
            checkpoint = source_root / task["checkpoint_relative_path"]
            _add_bytes(archive, task["checkpoint_relative_path"], checkpoint.read_bytes())
    spec["source_archive_filename"] = archive_path.name
    spec["source_archive_sha256"] = sha256_file(archive_path)
    spec["source_manifest_sha256"] = manifest_payload["manifest_sha256"]
    spec_path.write_text(
        json.dumps(spec, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    destination = tmp_path / "destination"
    fixed_source = destination / "src/tarca/stage2/training.py"
    fixed_source.parent.mkdir(parents=True)
    fixed_source.write_bytes(b"fixed-source")
    server_bundle = tmp_path / "server-bundle.tar.gz"
    server_bundle.write_bytes(b"fixed-server-bundle")
    return archive_path, server_bundle, spec_path, destination


def test_recovery_archive_restores_only_artifacts_and_uses_snapshot_database(
    tmp_path: Path,
) -> None:
    from tarca.stage2.recovery import Stage2RecoveryInputReceipt
    from tarca.stage2.recovery_archive import restore_stage2_recovery_archive

    archive, server_bundle, spec_path, destination = _synthetic_archive(tmp_path)

    receipt = restore_stage2_recovery_archive(
        destination,
        recovery_archive=archive,
        server_bundle=server_bundle,
        spec_path=spec_path,
    )

    assert (destination / "src/tarca/stage2/training.py").read_bytes() == b"fixed-source"
    assert (
        destination / "artifacts/stage2/runtime/store/marker.bin"
    ).read_bytes() == b"restored-artifact"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert sha256_file(destination / "artifacts/stage2/runtime/execution.sqlite3") == (
        spec["source_database_sha256"]
    )
    assert receipt["source_archive_sha256"] == spec["source_archive_sha256"]
    assert receipt["server_bundle_sha256"] == sha256_file(server_bundle)
    assert receipt["receipt_sha256"] == canonical_json_hash(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    Stage2RecoveryInputReceipt.model_validate_json(
        (
            destination / "artifacts/stage2/runtime/recovery_input_receipt.json"
        ).read_text(encoding="utf-8")
    )
