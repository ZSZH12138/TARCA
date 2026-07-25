from __future__ import annotations

import ast
import subprocess
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "tarca"
CONTRACT_INIT = SOURCE_ROOT / "contracts" / "__init__.py"

FORBIDDEN_TOP_LEVEL_MODULES = {
    "adapters",
    "data",
    "data_loaders",
    "das",
    "dro",
    "finance",
    "financial",
    "hooks",
    "models",
    "concepts",
    "interventions",
    "localization",
    "ot",
    "robustness",
    "scm",
    "training",
    "metrics",
}

EXPECTED_DEFINITION_SITES = {
    "CONTRACT_SCHEMA_VERSION": "src/tarca/contracts/version.py",
    "SplitPartition": "src/tarca/contracts/types.py",
    "RegimeRelation": "src/tarca/contracts/types.py",
    "InterventionKind": "src/tarca/contracts/types.py",
    "RunStatus": "src/tarca/contracts/types.py",
    "JSONScalar": "src/tarca/contracts/types.py",
    "JSONValue": "src/tarca/contracts/types.py",
    "JSONMetadata": "src/tarca/contracts/types.py",
    "WindowBatch": "src/tarca/contracts/data.py",
    "ForecastDistribution": "src/tarca/contracts/forecast.py",
    "ConceptBatch": "src/tarca/contracts/concepts.py",
    "InterventionSite": "src/tarca/contracts/interventions.py",
    "InterventionSpec": "src/tarca/contracts/interventions.py",
    "basis_orthonormality_tolerance": "src/tarca/contracts/interventions.py",
    "validate_spec_against_site": "src/tarca/contracts/interventions.py",
    "ForecastModelAdapter": "src/tarca/contracts/adapters.py",
    "DataSplitSummary": "src/tarca/contracts/manifests.py",
    "WindowContractSummary": "src/tarca/contracts/manifests.py",
    "InterventionPair": "src/tarca/contracts/manifests.py",
    "DataManifest": "src/tarca/contracts/manifests.py",
    "RunManifest": "src/tarca/contracts/manifests.py",
    "MetricRecord": "src/tarca/contracts/manifests.py",
    "validate_disjoint_window_partitions": "src/tarca/contracts/manifests.py",
    "validate_intervention_pair_partitions": "src/tarca/contracts/manifests.py",
    "ArtifactLayout": "src/tarca/contracts/artifacts.py",
    "metrics_by_regime_schema": "src/tarca/contracts/arrow_schemas.py",
    "predictions_schema": "src/tarca/contracts/arrow_schemas.py",
    "intervention_pairs_schema": "src/tarca/contracts/arrow_schemas.py",
    "validate_arrow_schema": "src/tarca/contracts/arrow_schemas.py",
}

GENERATED_OR_MODEL_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".joblib",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}

AUTHORIZED_NON_RUN_ARTIFACTS = {
    "artifacts/stage1/contracts/STAGE1_CONTRACT_IMPLEMENTATION_REPORT.md",
}

FORMAL_RUN_ARTIFACT_NAMES = {
    "config.yaml",
    "data_manifest.json",
    "environment.txt",
    "git_state.txt",
    "intervention_pairs.parquet",
    "metrics.json",
    "metrics_by_regime.parquet",
    "predictions.parquet",
    "stdout.log",
}

CACHE_OR_CHECKPOINT_PARTS = {
    ".cache",
    ".pytest_cache",
    "__pycache__",
    "cache",
    "caches",
    "checkpoint",
    "checkpoints",
}


def _top_level_definition_names(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for statement in tree.body:
        if isinstance(statement, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.append(statement.name)
        elif isinstance(statement, ast.Assign):
            names.extend(target.id for target in statement.targets if isinstance(target, ast.Name))
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            names.append(statement.target.id)
    return tuple(names)


def _tracked_files() -> tuple[str, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "-z"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(path.decode("utf-8") for path in completed.stdout.split(b"\0") if path)


def _forbidden_tracked_outputs(tracked_files: tuple[str, ...]) -> tuple[str, ...]:
    forbidden: list[str] = []
    for tracked_file in tracked_files:
        if tracked_file in AUTHORIZED_NON_RUN_ARTIFACTS:
            continue
        path = PurePosixPath(tracked_file)
        is_artifact = bool(path.parts) and path.parts[0] == "artifacts"
        if (
            path.suffix.lower() in GENERATED_OR_MODEL_SUFFIXES
            or CACHE_OR_CHECKPOINT_PARTS.intersection(path.parts)
            or (is_artifact and path.name in FORMAL_RUN_ARTIFACT_NAMES)
        ):
            forbidden.append(tracked_file)
    return tuple(sorted(forbidden))


def test_no_forbidden_stage1_top_level_modules_exist_under_src_tarca() -> None:
    actual_modules = {
        child.stem if child.is_file() and child.suffix == ".py" else child.name
        for child in SOURCE_ROOT.iterdir()
    }

    assert not FORBIDDEN_TOP_LEVEL_MODULES.intersection(actual_modules)


def test_core_contract_definitions_have_one_expected_source_site() -> None:
    actual_sites: dict[str, list[str]] = {
        definition_name: [] for definition_name in EXPECTED_DEFINITION_SITES
    }
    for source_file in sorted(SOURCE_ROOT.rglob("*.py")):
        relative_path = source_file.relative_to(REPOSITORY_ROOT).as_posix()
        for definition_name in _top_level_definition_names(source_file):
            if definition_name in actual_sites:
                actual_sites[definition_name].append(relative_path)

    assert actual_sites == {
        definition_name: [expected_path]
        for definition_name, expected_path in EXPECTED_DEFINITION_SITES.items()
    }
    assert all(
        CONTRACT_INIT.relative_to(REPOSITORY_ROOT).as_posix() not in sites
        for sites in actual_sites.values()
    )


def test_adapter_module_contains_only_the_static_protocol_boundary() -> None:
    adapter_path = SOURCE_ROOT / "contracts" / "adapters.py"
    tree = ast.parse(adapter_path.read_text(encoding="utf-8"), filename=str(adapter_path))
    classes = [statement for statement in tree.body if isinstance(statement, ast.ClassDef)]

    assert [class_definition.name for class_definition in classes] == ["ForecastModelAdapter"]
    protocol = classes[0]
    assert any(isinstance(base, ast.Name) and base.id == "Protocol" for base in protocol.bases)
    methods = [
        statement
        for statement in protocol.body
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    assert [method.name for method in methods] == [
        "adapter_name",
        "model_hash",
        "is_frozen",
        "predict_distribution",
        "list_intervention_sites",
        "capture",
        "intervene",
    ]
    assert all(
        len(statement.body) == 1
        and isinstance(statement.body[0], ast.Expr)
        and isinstance(statement.body[0].value, ast.Constant)
        and statement.body[0].value.value is Ellipsis
        for statement in methods
    )


def test_contracts_init_is_only_a_declarative_import_export_surface() -> None:
    tree = ast.parse(CONTRACT_INIT.read_text(encoding="utf-8"), filename=str(CONTRACT_INIT))
    docstring = ast.get_docstring(tree, clean=False)
    assert docstring
    assert len(docstring.splitlines()) <= 2

    statements = tree.body[1:]
    assert statements
    assert all(isinstance(statement, ast.ImportFrom | ast.Assign) for statement in statements)

    imports = [statement for statement in statements if isinstance(statement, ast.ImportFrom)]
    assignments = [statement for statement in statements if isinstance(statement, ast.Assign)]
    assert imports
    assert len(assignments) == 1
    assert all(statement.level == 1 and statement.module for statement in imports)
    assert all(alias.name != "*" for statement in imports for alias in statement.names)

    assignment = assignments[0]
    assert len(assignment.targets) == 1
    assert isinstance(assignment.targets[0], ast.Name)
    assert assignment.targets[0].id == "__all__"
    assert isinstance(assignment.value, ast.Tuple)
    assert all(
        isinstance(element, ast.Constant) and isinstance(element.value, str)
        for element in assignment.value.elts
    )


def test_no_stage1_run_artifact_cache_or_model_weight_is_tracked() -> None:
    assert _forbidden_tracked_outputs(_tracked_files()) == ()


def test_artifact_policy_allows_the_authorized_contract_report_without_requiring_it() -> None:
    tracked_files = (
        "artifacts/stage0/STAGE0_IMPLEMENTATION_REPORT.md",
        "artifacts/stage1/contracts/STAGE1_CONTRACT_IMPLEMENTATION_REPORT.md",
        "docs/engineering_note.md",
    )

    assert _forbidden_tracked_outputs(tracked_files) == ()


def test_artifact_policy_rejects_generated_run_cache_and_model_outputs() -> None:
    forbidden_files = (
        "artifacts/stage1/run-1/metrics.json",
        "artifacts/stage1/run-1/predictions.parquet",
        "checkpoints/model.pth",
        ".cache/generated.json",
    )

    assert set(_forbidden_tracked_outputs(forbidden_files)) == set(forbidden_files)
