from __future__ import annotations

from tarca.contracts import CONTRACT_SCHEMA_VERSION, PROTOCOL_ID


def test_stage1a_uses_frozen_contract_and_protocol_versions() -> None:
    assert CONTRACT_SCHEMA_VERSION == "1.0.0"
    assert PROTOCOL_ID == "TARCA-E2E-STAGE-PROTOCOL-2.0"
