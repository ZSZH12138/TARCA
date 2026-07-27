"""Contracts for the public, operator-facing Stage 0 documentation."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
STAGE0_SCOPE_PATH = PROJECT_ROOT / "docs" / "stage0_scope.md"
PROJECT_PLAN_PATH = PROJECT_ROOT / "docs" / "TARCA_项目计划书.md"
IMPLEMENTATION_PLAN_PATH = PROJECT_ROOT / "docs" / "TARCA_具体实施计划.md"
PREREGISTRATION_PATH = PROJECT_ROOT / "docs" / "preregistration_v0.md"
ASSUMPTION_LEDGER_PATH = PROJECT_ROOT / "docs" / "assumption_ledger.md"
NOVELTY_CLAIMS_PATH = PROJECT_ROOT / "docs" / "novelty_claims.md"
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
    readme = _read(README_PATH)
    assert "MANUAL_VERIFICATION_STAGE0.md" not in readme
    assert "TARCA_项目汇报书.md" not in readme


def test_curated_public_artifacts_exist_and_are_non_empty() -> None:
    for path in CURATED_ARTIFACTS:
        assert path.exists(), f"Missing curated public artifact: {path}"
        assert _read(path).strip(), f"Curated public artifact is empty: {path}"


def test_public_status_is_honest_and_consistent() -> None:
    required_stage_field = "Stage 0 status: COMPLETED_AND_FROZEN"
    required_research_field = "Research status: PARTIALLY_COMPLETED"

    report_path = CURATED_ARTIFACTS[0]
    for path in (README_PATH, STAGE0_SCOPE_PATH, report_path):
        stage_status_lines = [
            line.strip()
            for line in _read(path).splitlines()
            if line.strip().startswith("Stage 0 status:")
        ]
        assert stage_status_lines == [required_stage_field], (
            f"{path.relative_to(PROJECT_ROOT)} must contain exactly one formal completed-and-"
            "frozen Stage 0 delivery status."
        )

    for path in (README_PATH, report_path):
        research_status_lines = [
            line.strip()
            for line in _read(path).splitlines()
            if line.strip().startswith("Research status:")
        ]
        assert research_status_lines == [required_research_field], (
            f"{path.relative_to(PROJECT_ROOT)} must contain exactly one formal overall "
            "Research status field; later scientific stages remain incomplete."
        )


def test_stage1_engineering_status_is_honest_and_consistent() -> None:
    readme = _read(README_PATH)
    project_plan = _read(PROJECT_PLAN_PATH)
    implementation_plan = _read(IMPLEMENTATION_PLAN_PATH)
    required_statuses = (
        "Stage 1A status: COMPLETED",
        "Stage 1B status: COMPLETED_ENGINEERING",
        "Scientific status: ENGINEERING_SMOKE_ONLY",
    )

    for document in (readme, project_plan, implementation_plan):
        assert all(status in document for status in required_statuses)
        assert "Stage 1+ 尚未实施" not in document

    assert "Gate A/1/2" in readme
    assert "不构成正式科学验证" in readme


def test_stage1_plans_preserve_frozen_contract_boundaries() -> None:
    project_plan = _read(PROJECT_PLAN_PATH)
    implementation_plan = _read(IMPLEMENTATION_PLAN_PATH)
    preregistration = _read(PREREGISTRATION_PATH)
    assumption_ledger = _read(ASSUMPTION_LEDGER_PATH)
    novelty_claims = _read(NOVELTY_CLAIMS_PATH)
    required_shared_contracts = (
        "Stage 0 status: COMPLETED_AND_FROZEN",
        "Research status: PARTIALLY_COMPLETED",
        "Gate A",
        "Gate 3：预测收益（探索性）",
        "联合训练只能作为次级证据",
        "zero-refit",
        "TO_BE_FROZEN_BEFORE_FIRST_FORMAL_EXPERIMENT",
    )

    for plan in (project_plan, implementation_plan):
        assert all(contract in plan for contract in required_shared_contracts)
        assert (
            "金融压力测试属于 RQ5/验证层；其结果只是 Gate 3 探索性预测收益判断的一项输入"
        ) in plan

    assert "首次系统定义" not in project_plan
    assert "PLOT-guided DAS 是 `NOT_NOVEL` 基线" in project_plan
    assert "PLOT-guided DAS 是 `NOT_NOVEL` 基线" in implementation_plan
    assert implementation_plan.index("# 第七部分：金融压力测试") < implementation_plan.index(
        "Gate 3：预测收益（探索性）"
    )
    assert (
        project_plan.index("### Gate A：固定位置干预")
        < project_plan.index("### Gate 1：合成定位")
        < project_plan.index("### Gate 2：跨状态解释")
        < project_plan.index("### Gate 3：预测收益（探索性）")
        < project_plan.index("### Gate 4：论文完整性")
    )
    assert (
        implementation_plan.index("### 12.4 Gate A：固定位置干预门槛")
        < implementation_plan.index("### 17.6 Gate 1：合成定位与反空洞性")
        < implementation_plan.index("### 20.6 Gate 2")
        < implementation_plan.index("### 27.6 Gate 3：预测收益（探索性）")
        < implementation_plan.index("### 30.4 Gate 4")
    )

    assert "### Gate 3：预测收益（探索性）" in preregistration
    assert "联合训练只能作为次要模式另报" in assumption_ledger
    assert "解释器、位置、normalizer 和映射保持 zero-refit" in assumption_ledger
    assert "| N4 | PLOT 引导的 DAS | `NOT_NOVEL`" in novelty_claims
    assert "| N8 | 金融序列作为强非平稳压力测试 | `NOT_NOVEL`" in novelty_claims


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
