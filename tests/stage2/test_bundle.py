import json
import tarfile
from pathlib import Path

import pytest

from scripts.prepare_stage2_v1_server_bundle import build_stage2_server_bundle

ROOT = Path(__file__).resolve().parents[2]


def test_stage2_container_uses_the_locally_verified_prebuilt_frontend() -> None:
    dockerfile = (ROOT / "deploy/stage2/Dockerfile").read_text(encoding="utf-8")

    assert "FROM node:" not in dockerfile
    assert "npm ci" not in dockerfile
    assert (
        "COPY frontend/stage1b-monitor/dist/ "
        "/opt/tarca/frontend/stage1b-monitor/dist/"
    ) in dockerfile


def test_stage2_bundle_is_byte_deterministic(tmp_path: Path) -> None:
    first = build_stage2_server_bundle(ROOT, tmp_path / "a.tar.gz")
    second = build_stage2_server_bundle(ROOT, tmp_path / "b.tar.gz")
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert (tmp_path / "a.tar.gz").read_bytes() == (tmp_path / "b.tar.gz").read_bytes()
    assert first["formal_tasks_executed"] == 0


def test_stage2_bundle_has_offline_inputs_and_no_local_path_or_secret(tmp_path: Path) -> None:
    output = tmp_path / "bundle.tar.gz"
    receipt = build_stage2_server_bundle(ROOT, output)
    with tarfile.open(output, "r:gz") as archive:
        names = tuple(member.name for member in archive.getmembers())
        assert names == tuple(sorted(names))
        assert "SHA256SUMS.json" in names
        pyarrow_wheel = (
            "deploy/stage2/wheelhouse/"
            "pyarrow-25.0.1-cp310-cp310-manylinux_2_28_x86_64.whl"
        )
        assert pyarrow_wheel in names
        assert not any("pyarrow-20.0.0" in name for name in names)
        assert "artifacts/stage2/source-capsules/stage2-v1-official-sources.tar.gz" in names
        assert "docs/research/stage2_e02_local_implementation_report_v1.md" in names
        assert "docs/research/stage2_e02_server_handoff_v1.md" in names
        assert "docs/research/stage2_device_mismatch_recovery_v1.md" in names
        assert "configs/stage2/stage2_device_mismatch_recovery_v1.json" in names
        assert "src/tarca/stage2/recovery_archive.py" in names
        assert not any(name.startswith("third_party/") for name in names)
        for member in archive.getmembers():
            if member.isfile():
                handle = archive.extractfile(member)
                assert handle is not None
                payload = handle.read()
                assert b"C:" + b"\\Users" + b"\\DELL" not in payload
                assert b"-----BEGIN " + b"PRIVATE KEY-----" not in payload
    assert receipt["bundle_sha256"] == (tmp_path / "bundle.tar.gz.sha256").read_text().split()[0]
    decoded = json.loads((tmp_path / "bundle.tar.gz.receipt.json").read_text(encoding="utf-8"))
    assert decoded == receipt


def test_stage2_bundle_recovery_binding_rejects_the_wrong_archive(tmp_path: Path) -> None:
    wrong_archive = tmp_path / "tarca-stage2-recovery-20260831T102151Z.tar.gz"
    wrong_archive.write_bytes(b"not-the-frozen-recovery-archive")

    with pytest.raises(ValueError, match="recovery archive SHA-256"):
        build_stage2_server_bundle(
            ROOT,
            tmp_path / "bundle.tar.gz",
            recovery_archive=wrong_archive,
        )
