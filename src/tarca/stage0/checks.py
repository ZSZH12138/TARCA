from __future__ import annotations

import csv
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tarca.contracts import (
    ArtifactIndex,
    AuthorizedOverwriteReceipt,
    EnvironmentBundle,
    GateDecision,
    GateStatus,
    RelatedWorkBundle,
    ResearchContractManifest,
    Stage0CompletionReceipt,
    Stage0VerificationReport,
    sha256_file,
)

from .artifact_store import LocalArtifactStore
from .contract import freeze_research_contract, verify_research_contract
from .environment import run_doctor
from .sources import audit_dependency_bindings, load_sources_manifest

RELATED_WORK_COLUMNS = (
    "work_id",
    "title",
    "year",
    "venue_status",
    "paper_url",
    "problem",
    "intervention_type",
    "location_axes",
    "output_type",
    "robustness",
    "anti_injection",
    "code_url",
    "reusable_component",
    "gap_to_TARCA",
    "verification_date",
)

REQUIRED_WORK_IDS = frozenset(
    {
        "plot-2605.06979",
        "diroca-2510.04842",
        "transport-2608.15645",
        "cae-2607.00267",
        "hyperdas-2503.10894",
        "nonlinear-dilemma-2507.08802",
        "good-apples-2605.02234",
        "rep-divergence-2511.04638",
        "timesae-2601.09776",
        "chronos-sae-2603.10071",
        "ts-cbm-2410.06070",
        "forecastcf-2310.08137",
        "foil-2406.09130",
        "cogs-aaai-2026",
    }
)

REQUIRED_INPUTS = (
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "docs/stage0_scope.md",
    "docs/related_work_matrix.csv",
    "docs/novelty_claims.md",
    "docs/assumption_ledger.md",
    "docs/terminology.md",
    "docs/preregistration_v0.md",
    "third_party_manifest/sources.yaml",
)


def validate_related_work_matrix(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = set(RELATED_WORK_COLUMNS) - set(fieldnames)
        if missing:
            raise ValueError(f"related-work matrix missing columns: {sorted(missing)}")
        if fieldnames != RELATED_WORK_COLUMNS:
            raise ValueError("related-work matrix must use the exact columns in canonical order")
        rows = list(reader)
    work_ids = [row["work_id"] for row in rows]
    if any(not value for value in work_ids):
        raise ValueError("related-work matrix contains a blank work_id")
    if len(work_ids) != len(set(work_ids)):
        raise ValueError("related-work matrix contains duplicate work_id values")
    required_columns = set(RELATED_WORK_COLUMNS) - {"code_url"}
    for row_index, row in enumerate(rows, start=2):
        blank = sorted(column for column in required_columns if not (row[column] or "").strip())
        if blank:
            raise ValueError(
                f"related-work matrix row {row_index} has blank required value: {blank}"
            )
    return {"row_count": len(rows), "unique_work_ids": len(set(work_ids))}


def _matrix_work_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["work_id"] for row in csv.DictReader(handle)}


def _artifact_index(repo_root: Path, manifest: ResearchContractManifest) -> ArtifactIndex:
    store = LocalArtifactStore(repo_root)
    bundle = store.load_contract(manifest.related_work_ref, RelatedWorkBundle)
    environment_bundle = store.load_contract(manifest.environment_lock_ref, EnvironmentBundle)
    refs = [
        manifest.preregistration_ref,
        manifest.novelty_claims_ref,
        manifest.assumption_ledger_ref,
        manifest.terminology_ref,
        manifest.environment_lock_ref,
        manifest.related_work_ref,
        bundle.matrix_ref,
        bundle.third_party_versions_ref,
        environment_bundle.pyproject_ref,
        environment_bundle.lock_ref,
        environment_bundle.profile_ref,
    ]
    return ArtifactIndex(schema_version="1.0.0", artifacts=tuple(refs))


def _validate_inputs(repo_root: Path) -> dict[str, int]:
    missing = [relative for relative in REQUIRED_INPUTS if not (repo_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage 0 inputs: {missing}")
    matrix_summary = validate_related_work_matrix(repo_root / "docs/related_work_matrix.csv")
    missing_work = REQUIRED_WORK_IDS - _matrix_work_ids(repo_root / "docs/related_work_matrix.csv")
    if missing_work:
        raise ValueError(f"related-work matrix missing required work IDs: {sorted(missing_work)}")
    sources = load_sources_manifest(repo_root / "third_party_manifest/sources.yaml")
    dependency_summary = audit_dependency_bindings(repo_root, sources)
    return {
        **matrix_summary,
        **dependency_summary,
        "source_count": len(sources.sources),
    }


def freeze_stage0(
    repo_root: Path,
    *,
    created_at: datetime | None = None,
    allow_frozen_overwrite: bool = False,
    authorization_reason: str | None = None,
) -> ResearchContractManifest:
    repo_root = repo_root.resolve()
    _validate_inputs(repo_root)
    artifact_dir = repo_root / "artifacts/stage0"
    manifest_path = artifact_dir / "research_contract_manifest.json"
    archive_relative_path: str | None = None
    previous_manifest_hash: str | None = None
    previous_decision: GateDecision | None = None
    if manifest_path.is_file():
        if not allow_frozen_overwrite:
            raise FileExistsError(
                "Stage 0 is already frozen; use explicit overwrite authorization to replace it"
            )
        if authorization_reason is None or not authorization_reason.strip():
            raise ValueError("authorization_reason is required for frozen overwrite")
        previous_manifest_hash = sha256_file(manifest_path)
        decision_path = artifact_dir / "gate0_decision.json"
        if decision_path.is_file():
            previous_decision = GateDecision.model_validate_json(decision_path.read_bytes())
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        archive_name = f"{timestamp}-{previous_manifest_hash[:12]}"
        archive_dir = artifact_dir / "history" / archive_name
        archive_dir.mkdir(parents=True, exist_ok=False)
        for existing in tuple(artifact_dir.iterdir()):
            if existing.is_file():
                shutil.move(existing, archive_dir / existing.name)
        archive_relative_path = archive_dir.relative_to(repo_root).as_posix()

    manifest = freeze_research_contract(
        repo_root,
        created_at=created_at,
    )
    store = LocalArtifactStore(repo_root)
    store.publish_contract(
        _artifact_index(repo_root, manifest),
        artifact_id="stage0-artifact-index",
        artifact_type="ARTIFACT_INDEX",
        relative_path="artifacts/stage0/artifact_index.json",
        overwrite=True,
    )
    if previous_decision is not None:
        try:
            _verify_gate0_decision(repo_root, previous_decision)
        except (FileNotFoundError, ValueError):
            pass
        else:
            store.publish_contract(
                previous_decision,
                artifact_id="stage0-gate0-decision",
                artifact_type="GATE_DECISION",
                relative_path="artifacts/stage0/gate0_decision.json",
            )
    if previous_manifest_hash is not None:
        receipt = AuthorizedOverwriteReceipt(
            schema_version="1.0.0",
            action="AUTHORIZED_FROZEN_OVERWRITE",
            authorization_reason=authorization_reason or "",
            archived_previous_artifacts_at=archive_relative_path or "",
            previous_manifest_hash=previous_manifest_hash,
            replacement_manifest_hash=sha256_file(manifest_path),
        )
        store.publish_contract(
            receipt,
            artifact_id="stage0-authorized-overwrite-receipt",
            artifact_type="AUTHORIZED_OVERWRITE_RECEIPT",
            relative_path="artifacts/stage0/authorized_overwrite_receipt.json",
        )
    return manifest


def _verify_decision_evidence(repo_root: Path, decision: GateDecision) -> None:
    store = LocalArtifactStore(repo_root)
    for evidence in decision.evidence:
        store.verify_artifact(evidence)


def _verify_gate0_decision(repo_root: Path, decision: GateDecision) -> None:
    if decision.gate_id != "GATE_0_NOVELTY":
        raise ValueError("GateDecision gate_id is not GATE_0_NOVELTY")
    evidence_types = {item.artifact_type for item in decision.evidence}
    required_evidence = {"NOVELTY_CLAIMS", "RELATED_WORK_BUNDLE"}
    missing_evidence = required_evidence - evidence_types
    if missing_evidence:
        raise ValueError(f"Gate 0 decision is missing evidence: {sorted(missing_evidence)}")
    _verify_decision_evidence(repo_root, decision)
    if decision.status is not GateStatus.PASS:
        raise ValueError(f"Gate 0 is not PASS: {decision.status.value}")


def _verify_authorized_overwrite_receipt(repo_root: Path) -> None:
    relative_path = "artifacts/stage0/authorized_overwrite_receipt.json"
    receipt_path = repo_root / relative_path
    if not receipt_path.is_file():
        return
    store = LocalArtifactStore(repo_root)
    receipt_ref = store.ref_for_file(
        artifact_id="stage0-authorized-overwrite-receipt",
        artifact_type="AUTHORIZED_OVERWRITE_RECEIPT",
        relative_path=relative_path,
    )
    receipt = store.load_contract(receipt_ref, AuthorizedOverwriteReceipt)
    manifest_path = repo_root / "artifacts/stage0/research_contract_manifest.json"
    if sha256_file(manifest_path) != receipt.replacement_manifest_hash:
        raise ValueError("authorized overwrite receipt does not bind the replacement manifest")
    archive_dir = store.resolve(receipt.archived_previous_artifacts_at)
    if not archive_dir.is_dir():
        raise FileNotFoundError(archive_dir)
    archived_manifest = archive_dir / "research_contract_manifest.json"
    if not archived_manifest.is_file():
        raise FileNotFoundError(archived_manifest)
    if sha256_file(archived_manifest) != receipt.previous_manifest_hash:
        raise ValueError("authorized overwrite receipt does not bind the archived manifest")


def _verify_stage0_core(repo_root: Path, *, run_doctor_check: bool) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    store = LocalArtifactStore(repo_root)
    summary: dict[str, Any] = _validate_inputs(repo_root)
    manifest_ref = store.ref_for_file(
        artifact_id="stage0-research-contract",
        artifact_type="RESEARCH_CONTRACT_MANIFEST",
        relative_path="artifacts/stage0/research_contract_manifest.json",
    )
    manifest = store.load_contract(manifest_ref, ResearchContractManifest)
    verify_research_contract(manifest, repo_root)
    _verify_authorized_overwrite_receipt(repo_root)
    decision_ref = store.ref_for_file(
        artifact_id="stage0-gate0-decision",
        artifact_type="GATE_DECISION",
        relative_path="artifacts/stage0/gate0_decision.json",
    )
    decision = store.load_contract(decision_ref, GateDecision)
    _verify_gate0_decision(repo_root, decision)
    index_ref = store.ref_for_file(
        artifact_id="stage0-artifact-index",
        artifact_type="ARTIFACT_INDEX",
        relative_path="artifacts/stage0/artifact_index.json",
    )
    index = store.load_contract(index_ref, ArtifactIndex)
    if index != _artifact_index(repo_root, manifest):
        raise ValueError("artifact index does not match the frozen research contract")
    for artifact in index.artifacts:
        store.verify_artifact(artifact)
    if run_doctor_check:
        doctor = run_doctor(repo_root)
        if doctor.status != "PASS":
            raise RuntimeError(f"Stage 0 doctor failed: {doctor.error or 'unknown'}")
        summary["doctor"] = doctor
    summary.update(
        {
            "status": "PASS",
            "research_contract_status": manifest.status,
            "gate0_status": decision.status.value,
        }
    )
    return summary


def complete_stage0(
    repo_root: Path,
    *,
    completed_at: datetime | None = None,
    run_doctor_check: bool = True,
) -> Stage0CompletionReceipt:
    repo_root = repo_root.resolve()
    _verify_stage0_core(repo_root, run_doctor_check=run_doctor_check)
    store = LocalArtifactStore(repo_root)
    receipt = Stage0CompletionReceipt(
        schema_version="1.0.0",
        status="COMPLETED",
        completed_at=completed_at or datetime.now(UTC),
        research_contract_ref=store.ref_for_file(
            artifact_id="stage0-research-contract",
            artifact_type="RESEARCH_CONTRACT_MANIFEST",
            relative_path="artifacts/stage0/research_contract_manifest.json",
        ),
        gate_decision_ref=store.ref_for_file(
            artifact_id="stage0-gate0-decision",
            artifact_type="GATE_DECISION",
            relative_path="artifacts/stage0/gate0_decision.json",
        ),
        artifact_index_ref=store.ref_for_file(
            artifact_id="stage0-artifact-index",
            artifact_type="ARTIFACT_INDEX",
            relative_path="artifacts/stage0/artifact_index.json",
        ),
    )
    store.publish_contract(
        receipt,
        artifact_id="stage0-completion-receipt",
        artifact_type="STAGE_COMPLETION_RECEIPT",
        relative_path="artifacts/stage0/stage0_completion_receipt.json",
    )
    return receipt


def verify_stage0(repo_root: Path, *, run_doctor_check: bool = True) -> Stage0VerificationReport:
    repo_root = repo_root.resolve()
    summary = _verify_stage0_core(repo_root, run_doctor_check=run_doctor_check)
    store = LocalArtifactStore(repo_root)
    receipt_ref = store.ref_for_file(
        artifact_id="stage0-completion-receipt",
        artifact_type="STAGE_COMPLETION_RECEIPT",
        relative_path="artifacts/stage0/stage0_completion_receipt.json",
    )
    receipt = store.load_contract(receipt_ref, Stage0CompletionReceipt)
    expected_refs = {
        "research_contract_ref": store.ref_for_file(
            artifact_id="stage0-research-contract",
            artifact_type="RESEARCH_CONTRACT_MANIFEST",
            relative_path="artifacts/stage0/research_contract_manifest.json",
        ),
        "gate_decision_ref": store.ref_for_file(
            artifact_id="stage0-gate0-decision",
            artifact_type="GATE_DECISION",
            relative_path="artifacts/stage0/gate0_decision.json",
        ),
        "artifact_index_ref": store.ref_for_file(
            artifact_id="stage0-artifact-index",
            artifact_type="ARTIFACT_INDEX",
            relative_path="artifacts/stage0/artifact_index.json",
        ),
    }
    for field, expected in expected_refs.items():
        actual = getattr(receipt, field)
        if actual != expected:
            raise ValueError(f"Stage 0 completion receipt has stale {field}")
        store.verify_artifact(actual)
    return Stage0VerificationReport.model_validate({**summary, "completion_status": receipt.status})
