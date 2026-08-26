from __future__ import annotations


def passing_receipt(qualification_id: str = "stage1b-qualification-v2") -> dict[str, object]:
    return {
        "schema_version": "2.0.0",
        "qualification_id": qualification_id,
        "suite_id": "stage1b-worlds-v2",
        "source_manifest_sha256": "a" * 64,
        "source_commits": {"gvar": "b" * 40, "scoring_rules_l96": "c" * 40},
        "source_evidence_verified": True,
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
                "world_id": "lorenz96_f10_v2",
                "family_id": "lorenz96_single_scale",
                "role": "PRIMARY_MECHANISTIC",
                "status": "PASS",
                "selected_neural_adapter": "ITransformerReference",
            },
            {
                "world_id": "lorenz96_twoscale_v2",
                "family_id": "lorenz96_two_scale",
                "role": "PRIMARY_MECHANISTIC",
                "selected_neural_adapter": None,
                "status": "FAIL",
            },
        ],
        "suite_decision": {
            "status": "PASS",
            "passed_world_ids": ["lorenz96_f10_v2"],
            "failed_world_ids": ["lorenz96_twoscale_v2"],
            "primary_families": ["lorenz96_single_scale"],
            "failed_checks": [],
        },
        "failure_ledger": [],
    }
