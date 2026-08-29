from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest


def _evidence_io() -> ModuleType:
    try:
        return importlib.import_module("tarca.stage1b.evidence_io")
    except ModuleNotFoundError:
        pytest.fail("tarca.stage1b.evidence_io must provide shared evidence I/O")


def test_sha256_helpers_use_exact_bytes(tmp_path: Path) -> None:
    evidence_io = _evidence_io()
    payload = b"abc"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)

    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert evidence_io.sha256_bytes(payload) == expected
    assert evidence_io.sha256_file(source, chunk_size=2) == expected


def test_write_canonical_json_is_newline_terminated_and_sorted(tmp_path: Path) -> None:
    evidence_io = _evidence_io()
    target = tmp_path / "nested/evidence.json"

    payload = evidence_io.write_canonical_json(target, {"z": 1, "a": 2}, replace=False)

    assert payload == b'{"a":2,"z":1}\n'
    assert target.read_bytes() == payload
    with pytest.raises(FileExistsError):
        evidence_io.write_canonical_json(target, {"replacement": True}, replace=False)
    assert target.read_bytes() == payload


def test_failed_atomic_replace_preserves_prior_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence_io = _evidence_io()
    target = tmp_path / "evidence.json"
    original = evidence_io.write_canonical_json(target, {"version": 1}, replace=False)

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(evidence_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        evidence_io.write_canonical_json(target, {"version": 2}, replace=True)

    assert target.read_bytes() == original
    assert not tuple(tmp_path.glob(".*.tmp"))
