from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from tarca.contracts import (
    ArtifactRef,
    EnvironmentBundle,
    EnvironmentProfile,
    RelatedWorkBundle,
    ResearchContractManifest,
    validate_research_contract,
)

from .artifact_store import LocalArtifactStore
from .environment import capture_environment_profile
from .sources import load_sources_manifest

SCHEMA_VERSION = "1.0.0"
PROTOCOL_ID = "TARCA-E2E-STAGE-PROTOCOL-2.0"


def _artifact_ref(
    repo_root: Path,
    *,
    artifact_id: str,
    artifact_type: str,
    relative_path: str,
) -> ArtifactRef:
    return LocalArtifactStore(repo_root).ref_for_file(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        relative_path=relative_path,
        schema_version=SCHEMA_VERSION,
    )


def _build_related_work_bundle(repo_root: Path) -> ArtifactRef:
    matrix_ref = _artifact_ref(
        repo_root,
        artifact_id="stage0-related-work-matrix",
        artifact_type="RELATED_WORK_MATRIX",
        relative_path="docs/related_work_matrix.csv",
    )
    third_party_ref = _artifact_ref(
        repo_root,
        artifact_id="stage0-third-party-versions",
        artifact_type="THIRD_PARTY_VERSIONS",
        relative_path="third_party_manifest/sources.yaml",
    )
    bundle = RelatedWorkBundle(
        schema_version=SCHEMA_VERSION,
        matrix_ref=matrix_ref,
        third_party_versions_ref=third_party_ref,
    )
    return LocalArtifactStore(repo_root).publish_contract(
        bundle,
        artifact_id="stage0-related-work-bundle",
        artifact_type="RELATED_WORK_BUNDLE",
        relative_path="artifacts/stage0/related_work_bundle.json",
        overwrite=True,
    )


def _build_environment_bundle(repo_root: Path) -> ArtifactRef:
    pyproject_ref = _artifact_ref(
        repo_root,
        artifact_id="stage0-pyproject",
        artifact_type="PYPROJECT",
        relative_path="pyproject.toml",
    )
    lock_ref = _artifact_ref(
        repo_root,
        artifact_id="stage0-uv-lock",
        artifact_type="UV_LOCK",
        relative_path="uv.lock",
    )
    profile = capture_environment_profile(repo_root)
    profile_ref = LocalArtifactStore(repo_root).publish_contract(
        profile,
        artifact_id="stage0-environment-profile",
        artifact_type="ENVIRONMENT_PROFILE",
        relative_path="artifacts/stage0/environment_profile.json",
        overwrite=True,
    )
    bundle = EnvironmentBundle(
        schema_version=SCHEMA_VERSION,
        pyproject_ref=pyproject_ref,
        lock_ref=lock_ref,
        profile_ref=profile_ref,
        default_profile_id=profile.profile_id,
        profile_selection_policy="DEFAULT_WITH_RUNTIME_OVERRIDE",
    )
    return LocalArtifactStore(repo_root).publish_contract(
        bundle,
        artifact_id="stage0-environment-bundle",
        artifact_type="ENVIRONMENT_BUNDLE",
        relative_path="artifacts/stage0/environment_bundle.json",
        overwrite=True,
    )


def freeze_research_contract(
    repo_root: Path,
    *,
    created_at: datetime | None = None,
) -> ResearchContractManifest:
    repo_root = repo_root.resolve()
    manifest_path = repo_root / "artifacts/stage0/research_contract_manifest.json"
    if manifest_path.is_file():
        raise FileExistsError("Stage 0 research contract is already frozen")
    load_sources_manifest(repo_root / "third_party_manifest/sources.yaml")
    effective_created_at = created_at or datetime.now(UTC)
    manifest = ResearchContractManifest(
        schema_version=SCHEMA_VERSION,
        protocol_id=PROTOCOL_ID,
        preregistration_ref=_artifact_ref(
            repo_root,
            artifact_id="stage0-preregistration",
            artifact_type="PREREGISTRATION",
            relative_path="docs/preregistration_v0.md",
        ),
        novelty_claims_ref=_artifact_ref(
            repo_root,
            artifact_id="stage0-novelty-claims",
            artifact_type="NOVELTY_CLAIMS",
            relative_path="docs/novelty_claims.md",
        ),
        assumption_ledger_ref=_artifact_ref(
            repo_root,
            artifact_id="stage0-assumption-ledger",
            artifact_type="ASSUMPTION_LEDGER",
            relative_path="docs/assumption_ledger.md",
        ),
        terminology_ref=_artifact_ref(
            repo_root,
            artifact_id="stage0-terminology",
            artifact_type="TERMINOLOGY",
            relative_path="docs/terminology.md",
        ),
        environment_lock_ref=_build_environment_bundle(repo_root),
        related_work_ref=_build_related_work_bundle(repo_root),
        created_at=effective_created_at,
        status="FROZEN",
    )
    LocalArtifactStore(repo_root).publish_contract(
        manifest,
        artifact_id="stage0-research-contract",
        artifact_type="RESEARCH_CONTRACT_MANIFEST",
        relative_path="artifacts/stage0/research_contract_manifest.json",
    )
    return manifest


def _verify_artifact(repo_root: Path, artifact: ArtifactRef) -> None:
    LocalArtifactStore(repo_root).verify_artifact(artifact)


def verify_research_contract(
    manifest: ResearchContractManifest,
    repo_root: Path,
) -> None:
    validate_research_contract(manifest)
    refs = (
        manifest.preregistration_ref,
        manifest.novelty_claims_ref,
        manifest.assumption_ledger_ref,
        manifest.terminology_ref,
        manifest.environment_lock_ref,
        manifest.related_work_ref,
    )
    for artifact in refs:
        _verify_artifact(repo_root, artifact)

    store = LocalArtifactStore(repo_root)
    bundle = store.load_contract(manifest.related_work_ref, RelatedWorkBundle)
    for nested in (bundle.matrix_ref, bundle.third_party_versions_ref):
        _verify_artifact(repo_root, nested)

    environment_bundle = store.load_contract(manifest.environment_lock_ref, EnvironmentBundle)
    for nested in (
        environment_bundle.pyproject_ref,
        environment_bundle.lock_ref,
        environment_bundle.profile_ref,
    ):
        _verify_artifact(repo_root, nested)
    profile = store.load_contract(environment_bundle.profile_ref, EnvironmentProfile)
    if environment_bundle.default_profile_id != profile.profile_id:
        raise ValueError("environment bundle default profile does not match profile artifact")
    load_sources_manifest(repo_root / "third_party_manifest/sources.yaml")
