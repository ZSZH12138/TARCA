from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]  # PyArrow 25 ships no py.typed marker.

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tarca.artifacts.freeze import frozen_relative_paths  # noqa: E402
from tarca.contracts import CONTRACT_SCHEMA_VERSION, PROTOCOL_ID  # noqa: E402
from tarca.contracts.arrow_schemas import (  # noqa: E402
    EFFECTS_SCHEMA,
    INTERVENTION_PAIRS_SCHEMA,
    LOCALIZATION_SCHEMA,
    METRICS_SCHEMA,
    PREDICTIONS_SCHEMA,
    validate_table,
)
from tarca.stage0.checks import verify_stage0  # noqa: E402

_REQUIRED_FILES = frozenset(
    {
        "docs/stage1a_scope.md",
        "scripts/check_stage1a.py",
        "src/tarca/artifacts/store.py",
        "src/tarca/contracts/arrow_schemas.py",
        "src/tarca/contracts/data.py",
        "src/tarca/data/repository.py",
    }
)


def check_stage1a(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    root = repo_root.resolve()
    if CONTRACT_SCHEMA_VERSION != "1.0.0":
        raise ValueError("unexpected Stage 1A contract schema version")
    if PROTOCOL_ID != "TARCA-E2E-STAGE-PROTOCOL-2.0":
        raise ValueError("unexpected stable protocol identity")
    missing = sorted(path for path in _REQUIRED_FILES if not (root / path).is_file())
    if missing:
        raise FileNotFoundError(f"missing Stage 1A boundary files: {missing}")
    schemas = (
        PREDICTIONS_SCHEMA,
        INTERVENTION_PAIRS_SCHEMA,
        EFFECTS_SCHEMA,
        METRICS_SCHEMA,
        LOCALIZATION_SCHEMA,
    )
    for schema in schemas:
        validate_table(pa.Table.from_pylist([], schema=schema), schema)
    frozen = set(frozen_relative_paths(root))
    unfrozen = sorted(_REQUIRED_FILES - frozen)
    if unfrozen:
        raise ValueError(f"Stage 1A boundary files are absent from frozen catalog: {unfrozen}")
    stage0 = verify_stage0(root, run_doctor_check=False)
    return {
        "status": "PASS",
        "stage0_status": stage0.status,
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "arrow_schema_count": len(schemas),
        "frozen_path_count": len(frozen),
        "formal_data_touched": False,
        "training_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the TARCA Stage 1A boundary.")
    parser.add_argument("--json", action="store_true", help="Emit one JSON document.")
    args = parser.parse_args()
    try:
        result = check_stage1a()
    except Exception as exc:
        result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        exit_code = 1
    else:
        exit_code = 0
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Stage 1A check: {result['status']}")
        if "error" in result:
            print(f"  error: {result['error']}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
