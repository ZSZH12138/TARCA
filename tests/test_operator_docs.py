"""Contracts for the public, operator-facing Stage 0 documentation."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
CURATED_ARTIFACTS = (
    PROJECT_ROOT / "artifacts" / "stage0" / "STAGE0_IMPLEMENTATION_REPORT.md",
    PROJECT_ROOT / "artifacts" / "stage0" / "third_party_commits.json",
    PROJECT_ROOT / "artifacts" / "stage0" / "reference_smoke" / "plot" / "result_summary.md",
    PROJECT_ROOT / "artifacts" / "stage0" / "reference_smoke" / "diroca" / "result_summary.md",
)
CAUSAL_BOUNDARIES = (
    "TARCA 首篇研究只能对模型内部计算机制提出因果陈述。",
    "模型内部干预一致性不能自动推出真实金融市场中的因果关系。",
)
DIAGNOSTIC_STATUSES = ("PASS", "WARN", "SKIP", "FAIL")
REFERENCE_STATUSES = (
    "IMPORT_ONLY",
    "SMOKE_PASSED",
    "PARTIAL",
    "BLOCKED_BY_HARDWARE",
    "BLOCKED_BY_DEPENDENCY",
    "FAILED",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_readme_has_project_stage_boundaries_and_tree() -> None:
    readme = _read(README_PATH)

    assert "TARCA" in readme
    assert "Stage 0" in readme
    assert all(sentence in readme for sentence in CAUSAL_BOUNDARIES)
    for path in (
        "docs/",
        "src/tarca/stage0/",
        "scripts/",
        "tests/",
        "third_party_manifest/",
        "artifacts/stage0/",
    ):
        assert path in readme


def test_readme_documents_portable_windows_and_plain_uv_paths() -> None:
    readme = _read(README_PATH)

    assert "TARCA_CONDA_PREFIX" in readme
    assert r"$env:TARCA_CONDA_PREFIX" in readme
    assert r"$env:TARCA_CONDA_PREFIX\Scripts\uv.exe" in readme
    assert "UV_PROJECT_ENVIRONMENT" in readme
    assert "Linux/macOS/CI" in readme
    assert "uv sync --frozen --extra research --group dev" in readme
    assert ".venv" in readme


def test_readme_documents_operator_commands_and_statuses() -> None:
    readme = _read(README_PATH)

    required_commands = (
        "python scripts/doctor.py",
        "--json artifacts/stage0/doctor_report.json",
        "--markdown artifacts/stage0/doctor_report.md",
        "pytest",
        "--cov",
        "pre-commit",
        "make doctor",
        "make smoke",
        "make test",
        "scripts/run_reference_smoke.py plot",
        "scripts/run_reference_smoke.py diroca",
    )
    assert all(command in readme for command in required_commands)
    assert all(status in readme for status in DIAGNOSTIC_STATUSES)
    assert all(status in readme for status in REFERENCE_STATUSES)


def test_readme_states_hardware_third_party_and_stage1_limits() -> None:
    readme = _read(README_PATH)

    required_text = (
        "LOCAL_OK",
        "不需要 GPU",
        "固定 commit",
        "UNVERIFIED",
        "CPU-only",
        "Stage 1",
        "不实现",
        "IMPORT_ONLY",
        "PARTIAL",
        "不是论文复现",
        "MCQA",
        "Gemma",
        "Slurm",
        "金融数据",
        "SCM",
        "DAS",
        "DRO",
    )
    assert all(text in readme for text in required_text)


def test_readme_does_not_direct_operators_to_local_manual() -> None:
    assert "MANUAL_VERIFICATION_STAGE0.md" not in _read(README_PATH)


def test_curated_public_artifacts_exist_and_are_non_empty() -> None:
    for path in CURATED_ARTIFACTS:
        assert path.exists(), f"Missing curated public artifact: {path}"
        assert _read(path).strip(), f"Curated public artifact is empty: {path}"


def test_public_status_is_honest_and_consistent() -> None:
    required_field = "Research status: PARTIALLY_COMPLETED"

    readme_status_lines = [
        line.strip() for line in _read(README_PATH).splitlines() if line.strip() == required_field
    ]
    assert readme_status_lines == [required_field], (
        "README must contain exactly one formal Research status field with the approved "
        "incomplete Stage 0 value."
    )

    report = CURATED_ARTIFACTS[0]
    report_status_lines = [
        line.strip() for line in _read(report).splitlines() if line.strip() == required_field
    ]
    assert report_status_lines == [required_field], (
        "The public implementation report must contain exactly one matching formal "
        "Research status field."
    )


def test_operator_docs_do_not_claim_unperformed_scientific_success() -> None:
    combined = "\n".join((_read(README_PATH), *(_read(path) for path in CURATED_ARTIFACTS)))

    prohibited_claims = (
        "TARCA 已证明有效。",
        "TARCA 已完成机制定位。",
        "PLOT 已复现论文结果。",
        "DiRoCA 已复现论文结果。",
        "已证明真实金融市场因果关系。",
    )
    assert all(claim not in combined for claim in prohibited_claims)
