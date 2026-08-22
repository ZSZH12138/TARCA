from __future__ import annotations


def passing_receipt(qualification_id: str = "stage1b-qualification-v1") -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "qualification_id": qualification_id,
        "suite_id": "stage1b-worlds-v1",
        "source_commit": "a" * 40,
        "source_license_sha256": "b" * 64,
        "source_lock_verified": True,
        "world_config_sha256": "c" * 64,
        "qualification_config_sha256": "d" * 64,
        "hardware_receipt_sha256": "e" * 64,
        "partition_names": [
            "QUAL_TRAIN",
            "QUAL_TUNE",
            "QUAL_SEEN",
            "QUAL_UNSEEN",
        ],
        "experiment_ids": [],
        "world_decisions": [
            {
                "world_id": "network_cml_v1",
                "family_id": "nonlinear_network",
                "role": "PRIMARY_MECHANISTIC",
                "status": "PASS",
                "selected_neural_adapter": "SmallITransformer",
            },
            {
                "world_id": "ecology_lv_sde_v1",
                "family_id": "nonlinear_ecology",
                "role": "PRIMARY_MECHANISTIC",
                "status": "PASS",
                "selected_neural_adapter": "SmallPatchTST",
            },
        ],
        "suite_decision": {
            "status": "PASS",
            "passed_world_ids": ["network_cml_v1", "ecology_lv_sde_v1"],
            "failed_world_ids": [],
            "primary_families": ["nonlinear_network", "nonlinear_ecology"],
            "failed_checks": [],
        },
        "failure_ledger": [],
    }
