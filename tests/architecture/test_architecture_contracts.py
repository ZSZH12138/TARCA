from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
from pathlib import Path

import pytest

from tarca.architecture.skeleton.models import MechanisticModelAdapter
from tarca.contracts.adapters import ForecastPredictor
from tarca.contracts.architecture import ARCHITECTURE_VERSION
from tarca.contracts.common import (
    COMPLETED_TASK_POLICY,
    ArtifactRef,
    ScientificIdentity,
    TaskAttemptRecord,
    TaskState,
)
from tarca.contracts.effects import EffectSignature
from tarca.contracts.errors import (
    AuthorizationBlockedError,
    ErrorCode,
    UnimplementedCapabilityError,
)
from tarca.contracts.governance import SealedAccessGrant
from tarca.contracts.monitoring import MonitoringSnapshot

ROOT = Path(__file__).resolve().parents[2]


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts))


def test_no_duplicate_contract_definitions() -> None:
    registry = json.loads(
        (ROOT / "configs/architecture/contract_registry_v1.json").read_text(encoding="utf-8")
    )
    symbols = [entry["symbol"] for entry in registry["contracts"]]
    assert len(symbols) == len(set(symbols))
    assert all(
        entry["canonical_source"].startswith("src/tarca/contracts/")
        for entry in registry["contracts"]
    )


def test_current_public_api_compatibility() -> None:
    baseline = json.loads(
        (ROOT / "artifacts/architecture/CURRENT_PUBLIC_API_BASELINE.json").read_text(
            encoding="utf-8"
        )
    )
    for entry in baseline["symbols"]:
        module = importlib.import_module(entry["module"])
        assert hasattr(module, entry["symbol"]), entry
        actual_signature = str(inspect.signature(getattr(module, entry["symbol"])))

        def normalize(value: str) -> str:
            return re.sub(r"0x[0-9A-Fa-f]+", "0xADDR", value)

        assert normalize(actual_signature) == normalize(entry["signature"]), entry


def test_import_boundaries() -> None:
    forbidden_by_plane = {
        "src/tarca/contracts": {"tarca.training", "tarca.orchestration", "tarca.runtime"},
        "src/tarca/architecture/skeleton/monitoring": {
            "tarca.data",
            "tarca.models",
            "tarca.concepts",
            "tarca.interventions",
            "tarca.effects",
            "tarca.localization",
            "tarca.robustness",
            "tarca.metrics",
        },
    }
    for relative_root, forbidden in forbidden_by_plane.items():
        for path in _python_files(ROOT / relative_root):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = {
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module
            }
            imported.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            assert not any(
                any(name == item or name.startswith(f"{item}.") for name in imported)
                for item in forbidden
            ), path


def test_contract_roundtrip() -> None:
    artifact = ArtifactRef(
        artifact_id="artifact:model:one",
        artifact_type="checkpoint",
        content_hash="sha256:" + "a" * 64,
        schema_version="1",
        relative_path="artifacts/model.bin",
    )
    restored = ArtifactRef.from_mapping(artifact.to_mapping())
    assert restored == artifact
    identity = ScientificIdentity(
        protocol_id="protocol-v1",
        experiment_id="experiment-v1",
        task_id="task-v1",
        model_id="model-v1",
        data_id="data-v1",
        seed=7,
    )
    assert ScientificIdentity.from_mapping(identity.to_mapping()) == identity


def test_tensor_semantics_preserved() -> None:
    import torch

    tensor = torch.ones(2, 3, dtype=torch.float32)
    signature = EffectSignature(
        delta_mean=tensor,
        delta_scale=None,
        delta_quantiles={},
        horizon=3,
    )
    assert signature.delta_mean is tensor
    assert signature.delta_mean.dtype is torch.float32


def test_model_capability_separation() -> None:
    predictor_members = set(dir(ForecastPredictor))
    mechanistic_members = set(dir(MechanisticModelAdapter))
    assert "intervene" not in predictor_members
    assert ForecastPredictor not in MechanisticModelAdapter.__bases__
    assert "intervene" in mechanistic_members


def test_science_execution_separation() -> None:
    science_roots = (
        ROOT / "src/tarca/architecture/skeleton/data.py",
        ROOT / "src/tarca/architecture/skeleton/models.py",
        ROOT / "src/tarca/architecture/skeleton/concepts.py",
    )
    for root in science_roots:
        paths = (root,) if root.is_file() else _python_files(root)
        for path in paths:
            text = path.read_text(encoding="utf-8")
            assert "tarca.runtime" not in text
            assert "tarca.orchestration" not in text


def test_task_identity_attempt_independence() -> None:
    first = TaskAttemptRecord(
        task_id="task-1", attempt_id="attempt-1", attempt_number=1, state=TaskState.RUNNING
    )
    retry = TaskAttemptRecord(
        task_id="task-1", attempt_id="attempt-2", attempt_number=2, state=TaskState.RUNNING
    )
    assert first.task_id == retry.task_id
    assert first.attempt_id != retry.attempt_id
    assert first.attempt_number != retry.attempt_number


def test_completed_task_never_rerun() -> None:
    assert COMPLETED_TASK_POLICY == "NEVER_RERUN"
    assert TaskState.COMPLETED.value == "completed"


def test_append_only_artifact_contract() -> None:
    from tarca.architecture.skeleton.artifacts import ArtifactStore, publish_atomic

    assert "overwrite" not in dir(ArtifactStore)
    assert "delete" not in dir(ArtifactStore)
    with pytest.raises(UnimplementedCapabilityError) as error:
        publish_atomic(None, None)
    assert error.value.code is ErrorCode.UNIMPLEMENTED_CAPABILITY


def test_artifact_identity_independent_of_path() -> None:
    common = {
        "artifact_id": "artifact:metric:one",
        "artifact_type": "metric",
        "content_hash": "sha256:" + "b" * 64,
        "schema_version": "1",
    }
    left = ArtifactRef(**common, relative_path="runs/left/metric.json")
    right = ArtifactRef(**common, relative_path="runs/right/metric.json")
    assert left.identity_key == right.identity_key
    assert left.relative_path != right.relative_path


def test_monitor_is_read_only() -> None:
    public = {name for name in dir(MonitoringSnapshot) if not name.startswith("_")}
    assert not public.intersection({"write", "publish", "update", "delete", "materialize"})


def test_monitor_cannot_require_science_payload() -> None:
    fields = set(MonitoringSnapshot.__dataclass_fields__)
    assert fields == {
        "phase",
        "terminal_status",
        "task_counts",
        "resource_summary",
        "heartbeat_age_seconds",
        "eta_status",
    }


def test_sealed_requires_grant() -> None:
    from tarca.architecture.skeleton.governance import require_sealed_grant

    with pytest.raises(AuthorizationBlockedError) as error:
        require_sealed_grant(None)
    assert error.value.code is ErrorCode.AUTHORIZATION_BLOCKED
    grant = SealedAccessGrant(
        grant_id="grant-1", scope="phase9", protocol_hash="sha256:" + "c" * 64
    )
    assert require_sealed_grant(grant) == grant


def test_backend_types_do_not_escape() -> None:
    source = (ROOT / "src/tarca/architecture/skeleton/backends.py").read_text(encoding="utf-8")
    assert "import ot" not in source
    assert "import pyvene" not in source
    assert "from torch" not in source


def test_architecture_registry_consistency() -> None:
    module_registry = json.loads(
        (ROOT / "configs/architecture/module_registry_v1.json").read_text(encoding="utf-8")
    )
    dependency_rules = json.loads(
        (ROOT / "configs/architecture/dependency_rules_v1.json").read_text(encoding="utf-8")
    )
    names = {entry["name"] for entry in module_registry["modules"]}
    assert len(names) == len(module_registry["modules"])
    assert ARCHITECTURE_VERSION == module_registry["architecture_version"]
    for rule in dependency_rules["allowed_edges"]:
        assert rule["from"] in names
        assert rule["to"] in names


def test_contract_consumers_are_parallel_and_do_not_cross_implementations() -> None:
    dependency_rules = json.loads(
        (ROOT / "configs/architecture/dependency_rules_v1.json").read_text(encoding="utf-8")
    )
    allowed = {
        (rule["from"], rule["to"]) for rule in dependency_rules["allowed_edges"]
    }
    forbidden = {
        (rule["from"], rule["to"]) for rule in dependency_rules["forbidden_edges"]
    }
    consumers = {
        "data",
        "models",
        "concepts",
        "interventions",
        "effects",
        "localization",
        "robustness",
        "metrics",
        "experiments",
    }
    assert {("contracts", consumer) for consumer in consumers} <= allowed
    assert not allowed.intersection(
        {
            ("data", "models"),
            ("models", "data"),
            ("models", "concepts"),
            ("concepts", "models"),
        }
    )
    assert {
        ("data", "models"),
        ("models", "data"),
        ("models", "concepts"),
        ("concepts", "models"),
    } <= forbidden

    for path in (
        ROOT / "src/tarca/architecture/skeleton/models.py",
        ROOT / "src/tarca/architecture/skeleton/concepts.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            name.startswith("tarca.architecture.skeleton.data")
            or name.startswith("tarca.architecture.skeleton.models")
            or name.startswith("tarca.architecture.skeleton.concepts")
            for name in imported
        )


def test_unimplemented_capability_fails_closed() -> None:
    from tarca.architecture.skeleton.data import build_windows

    with pytest.raises(UnimplementedCapabilityError) as error:
        build_windows(None)
    assert error.value.code is ErrorCode.UNIMPLEMENTED_CAPABILITY
