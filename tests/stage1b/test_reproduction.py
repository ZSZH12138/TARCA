from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tarca.contracts import ArtifactRef
from tarca.stage1b.config import load_world_suite
from tarca.stage1b.reproduction import (
    ReproductionAdapter,
    ReproductionKind,
    ReproductionOutputs,
    ReproductionSpec,
    load_reproduction_suite,
    run_reproduction,
)
from tarca.stage1b.sources import MaterializedSources, materialize_source

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
OFFICIAL_BYTES = b"official source\n"
INPUT_BYTES = b'{"values":[1.0,2.0]}\n'


class FakeGit:
    def run(self, arguments: tuple[str, ...], cwd: Path | None = None) -> str:
        if arguments[:2] == ("checkout", "--detach"):
            assert cwd is not None
            (cwd / "official.py").write_bytes(OFFICIAL_BYTES)
        if arguments == ("rev-parse", "HEAD"):
            return COMMIT
        return ""


def _sources(tmp_path: Path) -> MaterializedSources:
    suite = load_world_suite(REPOSITORY_ROOT / "configs/stage1b/worlds_v2.yaml")
    template = suite.sources[0]
    source = template.model_copy(
        update={
            "source_id": "official",
            "repository_url": "https://example.org/official.git",
            "paper_url": "https://example.org/paper",
            "commit": COMMIT,
            "assets": (
                template.assets[0].model_copy(
                    update={
                        "asset_id": "generator",
                        "relative_path": "official.py",
                        "sha256": hashlib.sha256(OFFICIAL_BYTES).hexdigest(),
                    }
                ),
            ),
        }
    )
    receipt = materialize_source(source, tmp_path / "sources", FakeGit())
    return MaterializedSources(receipts=(receipt,))


def _spec(input_root: Path, **updates: object) -> ReproductionSpec:
    input_path = input_root / "input.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(INPUT_BYTES)
    payload: dict[str, object] = {
        "schema_version": "2.0.0",
        "case_id": "official-generator-parity",
        "kind": "GENERATOR",
        "source_id": "official",
        "source_commit": COMMIT,
        "asset_id": "generator",
        "adapter_key": "test_generator",
        "input_artifact": ArtifactRef(
            artifact_id="official-reproduction-input",
            artifact_type="official_reproduction_input",
            content_hash=hashlib.sha256(INPUT_BYTES).hexdigest(),
            schema_version="2.0.0",
            relative_path="input.json",
        ),
        "absolute_tolerance": 1e-10,
    }
    payload.update(updates)
    return ReproductionSpec.model_validate(payload)


def _adapter(adapter_values: tuple[float, ...] = (1.0, 2.0)) -> ReproductionAdapter:
    return ReproductionAdapter(
        adapter_key="test_generator",
        source_id="official",
        asset_id="generator",
        kind=ReproductionKind.GENERATOR,
        compare=lambda _root, _input: ReproductionOutputs(
            upstream=(1.0, 2.0),
            adapter=adapter_values,
        ),
    )


def test_reproduction_receipt_has_no_qualification_identity(tmp_path: Path) -> None:
    receipt = run_reproduction(
        _spec(tmp_path / "inputs"),
        _sources(tmp_path),
        adapters=(_adapter(),),
        input_root=tmp_path / "inputs",
    )

    payload = receipt.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True)
    assert receipt.channel == "OFFICIAL_REPRODUCTION"
    assert receipt.passed is True
    assert "qualification_id" not in payload
    assert "QUAL_UNSEEN" not in serialized


def test_reproduction_rejects_unregistered_asset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="registered adapter"):
        run_reproduction(
            _spec(tmp_path / "inputs", asset_id="other"),
            _sources(tmp_path),
            adapters=(_adapter(),),
            input_root=tmp_path / "inputs",
        )


def test_reproduction_rejects_input_hash_drift(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    spec = _spec(input_root)
    (input_root / "input.json").write_bytes(b"changed")

    with pytest.raises(ValueError, match="input artifact hash"):
        run_reproduction(
            spec,
            _sources(tmp_path),
            adapters=(_adapter(),),
            input_root=input_root,
        )


def test_reproduction_records_failed_numeric_parity(tmp_path: Path) -> None:
    receipt = run_reproduction(
        _spec(tmp_path / "inputs"),
        _sources(tmp_path),
        adapters=(_adapter((1.0, 2.1)),),
        input_root=tmp_path / "inputs",
    )

    assert receipt.maximum_absolute_error == pytest.approx(0.1)
    assert receipt.passed is False


@pytest.mark.parametrize("case_id", ["QUAL_UNSEEN-probe", "E01-probe", "e02-probe"])
def test_reproduction_identity_rejects_formal_namespaces(
    tmp_path: Path, case_id: str
) -> None:
    with pytest.raises(ValidationError, match="formal or qualification"):
        _spec(tmp_path / "inputs", case_id=case_id)


def test_official_reproduction_config_registers_six_isolated_cases() -> None:
    suite = load_reproduction_suite(
        REPOSITORY_ROOT / "configs/stage1b/official_reproduction_v2.yaml"
    )

    assert len(suite.cases) == 6
    assert {case.kind for case in suite.cases} == {
        ReproductionKind.GENERATOR,
        ReproductionKind.MODEL_FORWARD,
    }
    assert all(
        case.absolute_tolerance == (1e-10 if case.kind is ReproductionKind.GENERATOR else 1e-6)
        for case in suite.cases
    )
