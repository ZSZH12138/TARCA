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
        "qualification_seeds": [101, 103, 107],
        "reserved_formal_seeds": [201, 203],
        "qualification_evidence": {
            "official_source_receipt_sha256": "1" * 64,
            "reproduction_receipt_sha256": "2" * 64,
            "environment_receipt_sha256": "3" * 64,
            "precision_receipt_sha256": "4" * 64,
            "run_graph_sha256": "5" * 64,
            "task_manifest_sha256": "6" * 64,
            "execution_plan_sha256": "7" * 64,
            "hardware_receipt_sha256": "e" * 64,
            "completed_task_count": 74,
            "expected_task_count": 74,
        },
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
        "training_receipts": [],
        "comparisons": [],
        "suite_decision": {
            "status": "PASS",
            "passed_world_ids": ["lorenz96_f10_v2"],
            "failed_world_ids": ["lorenz96_twoscale_v2"],
            "primary_families": ["lorenz96_single_scale"],
            "failed_checks": [],
        },
        "failure_ledger": [],
    }
