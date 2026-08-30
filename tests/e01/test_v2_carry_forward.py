from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tarca.contracts import canonical_json_bytes, canonical_json_hash
from tarca.e01.v2_carry_forward import (
    CarryForwardVerificationError,
    verify_e01_b_carry_forward,
)
from tarca.e01.v2_config import E01V2Config, load_e01_v2_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/e01/e01_v2.yaml"
HISTORY = ROOT / "artifacts/e01/history/e01_v1_history_record.json"


def test_single_v1_history_record_proves_e01_b_without_hiding_v1_failure() -> None:
    config = load_e01_v2_config(CONFIG)

    assert config.carry_forward.report_path == config.carry_forward.recovery_validation_path
    assert ROOT / config.carry_forward.report_path == HISTORY

    receipt = verify_e01_b_carry_forward(ROOT, config)

    assert receipt.status == "PASS"
    assert receipt.v1_overall_gate_status == "FAIL"
    assert receipt.e01_b_convergence_seed_count == 5
    assert {item.control: item.seed_count for item in receipt.directional_seed_counts} == {
        "RANDOM_CONCEPT": 5,
        "WRONG_LAG": 5,
        "WRONG_SCM": 5,
    }
    assert receipt.e01_b_formal_seed_count == 5
    assert receipt.runtime_alert_count == 0
    assert len(receipt.receipt_sha256) == 64


def _copy_history(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    history = tmp_path / "e01_v1_history_record.json"
    history.write_bytes(HISTORY.read_bytes())
    payload = json.loads(history.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return history, payload


def _config_for(tmp_path: Path, history: Path) -> E01V2Config:
    config = load_e01_v2_config(CONFIG)
    payload = config.model_dump(mode="json")
    relative = history.relative_to(tmp_path).as_posix()
    payload["carry_forward"]["report_path"] = relative
    payload["carry_forward"]["recovery_validation_path"] = relative
    return E01V2Config.model_validate(payload)


def _rewrite_record(path: Path, record: dict[str, object]) -> None:
    unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
    sealed = {**unsigned, "record_sha256": canonical_json_hash(unsigned)}
    path.write_bytes(canonical_json_bytes(sealed) + b"\n")


def _mutate_embedded_json(
    record: dict[str, object],
    entry_name: str,
    mutate: Callable[[dict[str, Any]], None],
) -> str:
    evidence = record["original_evidence"]
    assert isinstance(evidence, dict)
    entry = evidence[entry_name]
    assert isinstance(entry, dict)
    raw = base64.b64decode(str(entry["content_base64"]), validate=True)
    payload = json.loads(raw)
    assert isinstance(payload, dict)
    mutate(payload)
    changed = canonical_json_bytes(payload) + b"\n"
    digest = hashlib.sha256(changed).hexdigest()
    entry["content_base64"] = base64.b64encode(changed).decode("ascii")
    entry["sha256"] = digest
    return digest


def test_history_record_rejects_changed_container_bytes(tmp_path: Path) -> None:
    history, _ = _copy_history(tmp_path)
    history.write_bytes(history.read_bytes() + b"\n")

    with pytest.raises(CarryForwardVerificationError, match="history record SHA-256"):
        verify_e01_b_carry_forward(tmp_path, _config_for(tmp_path, history))


def test_history_record_rejects_fabricated_l96_pass_count(tmp_path: Path) -> None:
    history, record = _copy_history(tmp_path)

    def lower_l96_count(payload: dict[str, Any]) -> None:
        payload["gate"]["convergence_seed_counts"]["lorenz96_twoscale_v2"] = 4

    report_hash = _mutate_embedded_json(record, "final_report", lower_l96_count)
    _rewrite_record(history, record)
    config = _config_for(tmp_path, history)
    payload = config.model_dump(mode="json")
    payload["carry_forward"]["report_sha256"] = report_hash

    with pytest.raises(CarryForwardVerificationError, match="convergence evidence"):
        verify_e01_b_carry_forward(tmp_path, E01V2Config.model_validate(payload))


def test_history_record_rejects_recovery_alerts(tmp_path: Path) -> None:
    history, record = _copy_history(tmp_path)

    def add_alert(payload: dict[str, Any]) -> None:
        payload["alert_count"] = 1

    recovery_hash = _mutate_embedded_json(record, "recovery_validation", add_alert)
    _rewrite_record(history, record)
    config = _config_for(tmp_path, history)
    payload = config.model_dump(mode="json")
    payload["carry_forward"]["recovery_validation_sha256"] = recovery_hash

    with pytest.raises(CarryForwardVerificationError, match="runtime alerts"):
        verify_e01_b_carry_forward(tmp_path, E01V2Config.model_validate(payload))
