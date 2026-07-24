from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tarca.stage0.sources import SourceEntry, load_sources  # noqa: E402

MANIFEST_PATH = PROJECT_ROOT / "third_party_manifest" / "sources.yaml"
REQUIRED_SOURCE_FIELDS = {
    "name",
    "paper_title",
    "paper_url",
    "repository_url",
    "role",
    "license",
    "default_branch",
    "verified_commit",
    "verified_at",
    "local_reference_path",
    "notes",
}


def test_manifest_entries_match_required_minimum_schema() -> None:
    assert MANIFEST_PATH.exists(), f"Missing manifest: {MANIFEST_PATH}"

    raw_manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw_manifest, list), "Manifest must be a YAML list."
    assert raw_manifest, "Manifest must contain at least one source entry."

    for index, entry in enumerate(raw_manifest):
        assert isinstance(entry, dict), f"Manifest entry {index} must be a mapping."
        assert set(entry) == REQUIRED_SOURCE_FIELDS, (
            f"Manifest entry {index} fields mismatch: {set(entry)}"
        )


def test_load_sources_returns_frozen_typed_entries() -> None:
    entries = load_sources(MANIFEST_PATH)

    assert len(entries) == 9
    assert all(isinstance(entry, SourceEntry) for entry in entries)
    with pytest.raises(ValidationError, match="frozen_instance"):
        entries[0].name = "changed"


def test_load_sources_rejects_invalid_yaml(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.yaml"
    manifest.write_text("- name: [unterminated", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_sources(manifest)


def test_load_sources_rejects_non_list_yaml(tmp_path: Path) -> None:
    manifest = tmp_path / "mapping.yaml"
    manifest.write_text("name: not-a-list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML list"):
        load_sources(manifest)


def test_load_sources_rejects_extra_fields(tmp_path: Path) -> None:
    raw_entry = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))[0]
    manifest = tmp_path / "extra.yaml"
    manifest.write_text(
        yaml.safe_dump([{**raw_entry, "unexpected": "value"}]),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        load_sources(manifest)


def test_load_sources_rejects_non_hex_verified_commit(tmp_path: Path) -> None:
    raw_entry = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))[0]
    manifest = tmp_path / "invalid-commit.yaml"
    manifest.write_text(
        yaml.safe_dump([{**raw_entry, "verified_commit": "g" * 40}]),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="40-character hexadecimal"):
        load_sources(manifest)


@pytest.mark.parametrize(
    "repository_url",
    [
        "--upload-pack=unsafe",
        "https://token@example.test/repository",
        "https://example.test/repository?token=secret",
    ],
)
def test_load_sources_rejects_unsafe_repository_url(tmp_path: Path, repository_url: str) -> None:
    raw_entry = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))[0]
    manifest = tmp_path / "unsafe-repository.yaml"
    manifest.write_text(
        yaml.safe_dump([{**raw_entry, "repository_url": repository_url}]),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="credential-free HTTPS URL"):
        load_sources(manifest)


@pytest.mark.parametrize(
    "local_reference_path",
    [
        "C:/outside/pyvene",
        ".cache/third_party/../pyvene",
        "third_party/pyvene",
        ".cache/third_party/not-pyvene",
    ],
    ids=["absolute", "traversal", "other-directory", "wrong-source-name"],
)
def test_load_sources_rejects_unsafe_local_reference_path(
    tmp_path: Path, local_reference_path: str
) -> None:
    raw_entry = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))[0]
    manifest = tmp_path / "unsafe-local-path.yaml"
    manifest.write_text(
        yaml.safe_dump([{**raw_entry, "local_reference_path": local_reference_path}]),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match=r"\.cache/third_party/<name>"):
        load_sources(manifest)


def test_manifest_contains_researched_metadata() -> None:
    entries = {entry.name: entry for entry in load_sources(MANIFEST_PATH)}

    assert entries["hyperdas"].paper_url == "https://arxiv.org/abs/2503.10894"
    assert entries["time_series_library"].paper_title == "UNRESOLVED"
    assert entries["time_series_library"].paper_url == "UNRESOLVED"
    assert entries["time_series_library"].license == "MIT"
    assert entries["patchtst"].paper_title == (
        "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
    )
    assert entries["patchtst"].paper_url == "https://arxiv.org/abs/2211.14730"
    assert entries["patchtst"].license == "Apache-2.0"
    assert entries["itransformer"].paper_title == (
        "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
    )
    assert entries["itransformer"].paper_url == "https://arxiv.org/abs/2310.06625"
    assert entries["itransformer"].license == "MIT"
    assert entries["chronos_forecasting"].paper_title == (
        "Chronos: Learning the Language of Time Series"
    )
    assert entries["chronos_forecasting"].paper_url == (
        "https://openreview.net/forum?id=gerNCVqqtR"
    )
    assert entries["chronos_forecasting"].license == "Apache-2.0"
    assert {entries[name].license for name in ("plot", "diroca", "hyperdas")} == {"UNVERIFIED"}
