from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from tarca.contracts.artifacts import ArtifactLayout
from tarca.contracts.manifests import StrictContractModel
from tarca.contracts.version import CONTRACT_SCHEMA_VERSION

EXPERIMENT_ID = "experiment-1"
RUN_ID = "run-1"
RUN_ROOT = PurePosixPath("artifacts/experiment-1/run-1")
REQUIRED_RELATIVE_PATHS = (
    RUN_ROOT / "config.yaml",
    RUN_ROOT / "metrics.json",
    RUN_ROOT / "metrics_by_regime.parquet",
    RUN_ROOT / "predictions.parquet",
    RUN_ROOT / "intervention_pairs.parquet",
    RUN_ROOT / "data_manifest.json",
    RUN_ROOT / "environment.txt",
    RUN_ROOT / "git_state.txt",
    RUN_ROOT / "stdout.log",
    RUN_ROOT / "plots",
)


def _layout(**overrides: object) -> ArtifactLayout:
    values: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "run_id": RUN_ID,
    }
    return ArtifactLayout(**{**values, **overrides})


def _make_directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr or completed.stdout
        return
    link.symlink_to(target, target_is_directory=True)


def test_layout_is_a_strict_persistent_contract_with_stable_fields() -> None:
    layout = _layout()

    assert isinstance(layout, StrictContractModel)
    assert tuple(ArtifactLayout.model_fields) == (
        "schema_version",
        "experiment_id",
        "run_id",
    )
    assert layout.schema_version == CONTRACT_SCHEMA_VERSION
    assert (
        ArtifactLayout.model_json_schema()["properties"]["schema_version"]["const"]
        == CONTRACT_SCHEMA_VERSION
    )


def test_layout_supports_strict_json_round_trip_and_rejects_mutation_or_extra_fields() -> None:
    layout = _layout()

    assert ArtifactLayout.model_validate_json(layout.model_dump_json()) == layout

    extra_payload = {**layout.model_dump(mode="json"), "unexpected": "value"}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ArtifactLayout.model_validate_json(json.dumps(extra_payload))

    wrong_version = {**layout.model_dump(mode="json"), "schema_version": "2.0.0"}
    with pytest.raises(ValidationError, match="schema_version"):
        ArtifactLayout.model_validate_json(json.dumps(wrong_version))

    with pytest.raises(ValidationError, match="frozen_instance"):
        layout.run_id = "replacement"


@pytest.mark.parametrize("field_name", ["experiment_id", "run_id"])
@pytest.mark.parametrize("value", ["", "   ", 7, None])
def test_layout_requires_strict_non_empty_identifiers(field_name: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field_name):
        _layout(**{field_name: value})


@pytest.mark.parametrize("field_name", ["experiment_id", "run_id"])
@pytest.mark.parametrize(
    "value",
    [
        "nested/segment",
        r"nested\segment",
        ".",
        "..",
        "C:",
        "C:drive-relative",
        "name:alternate-stream",
    ],
)
def test_layout_rejects_identifiers_that_are_not_safe_single_path_segments(
    field_name: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError, match=field_name):
        _layout(**{field_name: value})


def test_layout_exposes_the_exact_logical_root_and_required_relative_paths() -> None:
    layout = _layout()

    assert layout.relative_run_root == RUN_ROOT
    assert layout.required_relative_paths == REQUIRED_RELATIVE_PATHS
    assert all(not path.is_absolute() for path in layout.required_relative_paths)


def test_layout_validates_required_paths_and_additional_plot_descendants() -> None:
    layout = _layout()

    for expected in REQUIRED_RELATIVE_PATHS:
        assert layout.validate_relative_path(expected.as_posix()) == expected

    plot_path = RUN_ROOT / "plots" / "diagnostic.png"
    assert layout.validate_relative_path(plot_path.as_posix()) == plot_path
    assert layout.validate_relative_path(RUN_ROOT.as_posix()) == RUN_ROOT


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        "/artifacts/experiment-1/run-1/config.yaml",
        "//server/share/artifacts/experiment-1/run-1/config.yaml",
        r"C:\artifacts\experiment-1\run-1\config.yaml",
        "C:/artifacts/experiment-1/run-1/config.yaml",
        r"C:artifacts\experiment-1\run-1\config.yaml",
        r"\\server\share\artifacts\experiment-1\run-1\config.yaml",
        r"\\?\C:\artifacts\experiment-1\run-1\config.yaml",
        "artifacts//experiment-1/run-1/config.yaml",
        "artifacts/experiment-1/run-1/config.yaml/",
        "artifacts/./experiment-1/run-1/config.yaml",
        "artifacts/experiment-1/./run-1/config.yaml",
        "artifacts/experiment-1/run-1/./config.yaml",
        "artifacts/experiment-1/run-1/../other.json",
        "artifacts/experiment-1/../run-1/config.yaml",
        "../artifacts/experiment-1/run-1/config.yaml",
        r"artifacts/experiment-1/run-1\config.yaml",
        "outside/experiment-1/run-1/config.yaml",
        "artifacts/other-experiment/run-1/config.yaml",
        "artifacts/experiment-1/other-run/config.yaml",
    ],
)
def test_layout_rejects_malformed_or_out_of_layout_relative_paths(unsafe_path: str) -> None:
    with pytest.raises(ValueError, match="relative path"):
        _layout().validate_relative_path(unsafe_path)


@pytest.mark.parametrize("unsafe_path", [None, 7, Path("artifacts/experiment-1/run-1")])
def test_layout_requires_an_unmodified_string_for_relative_path_validation(
    unsafe_path: object,
) -> None:
    with pytest.raises(TypeError, match="relative_path"):
        _layout().validate_relative_path(unsafe_path)  # type: ignore[arg-type]


def test_resolve_path_returns_a_contained_absolute_path_without_creating_artifacts(
    tmp_path: Path,
) -> None:
    layout = _layout()
    relative_path = "artifacts/experiment-1/run-1/predictions.parquet"

    resolved = layout.resolve_path(tmp_path, relative_path)

    assert resolved == tmp_path.resolve() / Path(*PurePosixPath(relative_path).parts)
    assert resolved.is_absolute()
    assert not resolved.exists()
    assert not (tmp_path / "artifacts").exists()


def test_resolve_path_allows_existing_non_link_parent_directories(tmp_path: Path) -> None:
    run_root = tmp_path / "artifacts" / EXPERIMENT_ID / RUN_ID
    run_root.mkdir(parents=True)

    resolved = _layout().resolve_path(
        tmp_path,
        "artifacts/experiment-1/run-1/plots/diagnostic.png",
    )

    assert resolved == run_root / "plots" / "diagnostic.png"
    assert not resolved.exists()


def test_resolve_path_rejects_unsafe_lexical_input_before_filesystem_resolution(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside.json"

    with pytest.raises(ValueError, match="relative path"):
        _layout().resolve_path(
            tmp_path,
            "artifacts/experiment-1/run-1/../../../../outside.json",
        )

    assert not outside.exists()


def test_resolve_path_rejects_an_existing_parent_link_even_when_it_targets_inside_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "artifacts" / EXPERIMENT_ID / RUN_ID
    run_root.mkdir(parents=True)
    real_plots = tmp_path / "real-plots"
    real_plots.mkdir()
    _make_directory_link(run_root / "plots", real_plots)

    with pytest.raises(ValueError, match=r"symlink|junction|reparse"):
        _layout().resolve_path(
            tmp_path,
            "artifacts/experiment-1/run-1/plots/diagnostic.png",
        )


def test_resolve_path_rejects_an_existing_parent_link_that_escapes_root(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    try:
        _make_directory_link(tmp_path / "artifacts", outside)

        with pytest.raises(ValueError, match=r"symlink|junction|reparse"):
            _layout().resolve_path(
                tmp_path,
                "artifacts/experiment-1/run-1/config.yaml",
            )
    finally:
        outside.rmdir()


def test_resolve_path_rejects_a_link_used_as_the_caller_supplied_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    _make_directory_link(linked_root, real_root)

    with pytest.raises(ValueError, match=r"filesystem_root.*symlink|junction|reparse"):
        _layout().resolve_path(
            linked_root,
            "artifacts/experiment-1/run-1/config.yaml",
        )


def test_resolve_path_rejects_a_link_in_the_caller_root_ancestor_chain(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    filesystem_root = real_parent / "root"
    filesystem_root.mkdir(parents=True)
    linked_parent = tmp_path / "linked-parent"
    _make_directory_link(linked_parent, real_parent)

    with pytest.raises(ValueError, match=r"filesystem_root.*symlink|junction|reparse"):
        _layout().resolve_path(
            linked_parent / "root",
            "artifacts/experiment-1/run-1/config.yaml",
        )
