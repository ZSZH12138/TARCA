from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tarca.contracts import GateDecision, GateStatus, StrictContractModel, canonical_json_bytes
from tarca.stage0.checks import (
    complete_stage0,
    freeze_stage0,
    validate_related_work_matrix,
    verify_stage0,
)


def _write_approved_gate0(repo_root: Path, manifest: object) -> None:
    novelty_ref = manifest.novelty_claims_ref  # type: ignore[attr-defined]
    related_ref = manifest.related_work_ref  # type: ignore[attr-defined]
    decision = GateDecision(
        gate_id="GATE_0_NOVELTY",
        status=GateStatus.PASS,
        rationale=(
            "Human-authorized novelty review accepted the narrowed TARCA claims at the "
            "2026-08-20 evidence cutoff."
        ),
        evidence=(novelty_ref, related_ref),
    )
    path = repo_root / "artifacts/stage0/gate0_decision.json"
    path.write_bytes(canonical_json_bytes(decision.model_dump(mode="json")) + b"\n")


def test_validate_related_work_matrix(stage0_repo: Path) -> None:
    summary = validate_related_work_matrix(stage0_repo / "docs/related_work_matrix.csv")

    assert summary["row_count"] == 14
    assert summary["unique_work_ids"] == 14


def test_related_work_matrix_rejects_missing_column(stage0_repo: Path) -> None:
    path = stage0_repo / "docs/related_work_matrix.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("verification_date", "wrong_column", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing columns"):
        validate_related_work_matrix(path)


def test_related_work_matrix_rejects_extra_column(stage0_repo: Path) -> None:
    path = stage0_repo / "docs/related_work_matrix.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] += ",unexpected"
    lines[1:] = [f"{line},value" for line in lines[1:]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact columns"):
        validate_related_work_matrix(path)


def test_related_work_matrix_rejects_blank_required_value(stage0_repo: Path) -> None:
    path = stage0_repo / "docs/related_work_matrix.csv"
    path.write_text(
        path.read_text(encoding="utf-8").replace("PLOT,2026", ",2026", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="blank required value"):
        validate_related_work_matrix(path)


def test_freeze_and_verify_full_stage0_flow(stage0_repo: Path) -> None:
    manifest = freeze_stage0(
        stage0_repo,
        created_at=datetime(2026, 8, 20, 1, 2, 3, tzinfo=UTC),
    )
    _write_approved_gate0(stage0_repo, manifest)
    complete_stage0(
        stage0_repo,
        completed_at=datetime(2026, 8, 20, 2, 3, 4, tzinfo=UTC),
        run_doctor_check=False,
    )
    summary = verify_stage0(stage0_repo, run_doctor_check=False)

    assert manifest.status == "FROZEN"
    assert isinstance(summary, StrictContractModel)
    assert type(summary).__name__ == "Stage0VerificationReport"
    assert summary.status == "PASS"
    assert summary.gate0_status == "PASS"
    assert summary.completion_status == "COMPLETED"


def test_verify_requires_preapproved_gate0_decision(stage0_repo: Path) -> None:
    freeze_stage0(stage0_repo)

    with pytest.raises(FileNotFoundError):
        verify_stage0(stage0_repo, run_doctor_check=False)


def test_verify_requires_completion_receipt(stage0_repo: Path) -> None:
    manifest = freeze_stage0(stage0_repo)
    _write_approved_gate0(stage0_repo, manifest)

    with pytest.raises(FileNotFoundError):
        verify_stage0(stage0_repo, run_doctor_check=False)

    receipt = complete_stage0(stage0_repo, run_doctor_check=False)

    assert receipt.status == "COMPLETED"
    assert verify_stage0(stage0_repo, run_doctor_check=False).status == "PASS"


def test_stage0_freeze_is_default_frozen_but_explicitly_overridable(stage0_repo: Path) -> None:
    manifest = freeze_stage0(stage0_repo)
    _write_approved_gate0(stage0_repo, manifest)
    complete_stage0(stage0_repo, run_doctor_check=False)

    with pytest.raises(FileExistsError, match="already frozen"):
        freeze_stage0(stage0_repo)
    with pytest.raises(ValueError, match="authorization_reason"):
        freeze_stage0(stage0_repo, allow_frozen_overwrite=True)

    replacement = freeze_stage0(
        stage0_repo,
        allow_frozen_overwrite=True,
        authorization_reason="User approved rebuilding Stage 0 artifacts after contract repair.",
    )

    receipt = stage0_repo / "artifacts/stage0/authorized_overwrite_receipt.json"
    history = stage0_repo / "artifacts/stage0/history"
    assert receipt.is_file()
    assert any(history.iterdir())
    assert (stage0_repo / "artifacts/stage0/gate0_decision.json").is_file()
    assert not (stage0_repo / "artifacts/stage0/stage0_completion_receipt.json").exists()
    complete_stage0(stage0_repo, run_doctor_check=False)
    assert verify_stage0(stage0_repo, run_doctor_check=False).research_contract_status == (
        replacement.status
    )


def test_authorized_overwrite_receipt_must_bind_replacement_manifest(
    stage0_repo: Path,
) -> None:
    manifest = freeze_stage0(stage0_repo)
    _write_approved_gate0(stage0_repo, manifest)
    complete_stage0(stage0_repo, run_doctor_check=False)
    freeze_stage0(
        stage0_repo,
        allow_frozen_overwrite=True,
        authorization_reason="User approved replacing the frozen Stage 0 artifacts.",
    )
    receipt_path = stage0_repo / "artifacts/stage0/authorized_overwrite_receipt.json"
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["replacement_manifest_hash"] = "0" * 64
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="replacement manifest"):
        complete_stage0(stage0_repo, run_doctor_check=False)


def test_authoritative_docs_define_the_human_gate0_exception() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    protocol = (
        repo_root / "docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md"
    ).read_text(encoding="utf-8")
    implementation = (repo_root / "docs/auth/TARCA_具体实施计划.md").read_text(encoding="utf-8")
    project_plan = (repo_root / "docs/auth/TARCA_项目计划书.md").read_text(encoding="utf-8")
    terminology = (repo_root / "docs/terminology.md").read_text(encoding="utf-8")

    exception = "GATE_0_NOVELTY 是人工新颖性 Gate，不要求 GateSpec"  # noqa: RUF001
    assert exception in protocol
    assert exception in implementation
    assert exception in project_plan
    assert exception in terminology


def test_protocol_names_stage0_public_report_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    protocol = (
        repo_root / "docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md"
    ).read_text(encoding="utf-8")

    assert "run_doctor(workspace: PathLike) -> DoctorReport" in protocol
    assert "verify_stage0(...) -> Stage0VerificationReport" in protocol


def test_protocol_makes_software_tests_recommended_not_mandatory() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    protocol = (
        repo_root / "docs/auth/TARCA_END_TO_END_STAGE_PROTOCOL_SPECIFICATION_V2_0.md"
    ).read_text(encoding="utf-8")

    assert "因此必须配套静态类型检查和行为测试" not in protocol
    assert "行为测试必须验证" not in protocol
    assert "## 22.5 Wasserstein solver 单元测试（推荐）" in protocol  # noqa: RUF001
