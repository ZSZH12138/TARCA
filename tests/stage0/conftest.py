from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def stage0_repo(tmp_path: Path) -> Path:
    files = {
        "README.md": "# TARCA\n",
        "pyproject.toml": '[project]\nname = "tarca-test"\n',
        "docs/stage0_scope.md": "# scope\n",
        "docs/preregistration_v0.md": "# preregistration\n",
        "docs/novelty_claims.md": """# claims
<!-- TARCA_NOVELTY_CLAIMS_YAML_BEGIN -->
```yaml
schema_version: "1.0.0"
verification_date: "2026-08-20"
claims:
  - claim_id: TARCA-C1
    status: PROVISIONAL
    nearest_work: [plot-2605.06979]
    falsification: held-out horizon-by-lag test
    failure_action: DROP_CLAIM
  - claim_id: TARCA-C2
    status: PROVISIONAL
    nearest_work: [plot-2605.06979]
    falsification: joint truth recovery test
    failure_action: DROP_CLAIM
  - claim_id: TARCA-C3
    status: PROVISIONAL
    nearest_work: [diroca-2510.04842]
    falsification: sequential unseen zero-refit test
    failure_action: DROP_CLAIM
  - claim_id: TARCA-C4
    status: REQUIRED_SUPPORTING_CONTRIBUTION
    nearest_work: [cae-2607.00267]
    falsification: random and unmapped-variable controls
    failure_action: REPAIR
excluded_claims:
  - claim_id: TARCA-N1
    status: NOT_NOVEL
    nearest_work: [plot-2605.06979]
    failure_action: KEEP_AS_BASELINE
  - claim_id: TARCA-N2
    status: NOT_NOVEL
    nearest_work: [plot-2605.06979]
    failure_action: KEEP_AS_BASELINE
  - claim_id: TARCA-N3
    status: NOT_NOVEL
    nearest_work: [diroca-2510.04842]
    failure_action: KEEP_AS_BASELINE
  - claim_id: TARCA-N4
    status: NOT_NOVEL
    nearest_work: [cae-2607.00267]
    failure_action: KEEP_AS_BASELINE
  - claim_id: TARCA-N5
    status: NOT_NOVEL
    nearest_work: [timesae-2601.09776]
    failure_action: KEEP_AS_BASELINE
  - claim_id: TARCA-N6
    status: DROP_CLAIM
    nearest_work: []
    failure_action: STOP
```
<!-- TARCA_NOVELTY_CLAIMS_YAML_END -->
""",
        "docs/assumption_ledger.md": "# assumptions\n",
        "docs/terminology.md": "# terms\n",
        "docs/related_work_matrix.csv": (
            "work_id,title,year,venue_status,paper_url,problem,intervention_type,"
            "location_axes,output_type,robustness,anti_injection,code_url,"
            "reusable_component,gap_to_TARCA,verification_date\n"
            "plot-2605.06979,PLOT,2026,arXiv,https://arxiv.org/abs/2605.06979,"
            "localization,swap,layer,output,no,no,https://github.com/example/repo,"
            "baseline,gap,2026-08-20\n"
            "diroca-2510.04842,DiRoCA,2025,arXiv,https://arxiv.org/abs/2510.04842,"
            "robust abstraction,intervention,map,distribution,yes,no,,baseline,gap,"
            "2026-08-20\n"
            "cae-2607.00267,CAE,2026,workshop,https://arxiv.org/abs/2607.00267,"
            "metrics,intervention,variables,error,yes,yes,,metric,gap,2026-08-20\n"
            "timesae-2601.09776,TimeSAE,2026,conference,"
            "https://arxiv.org/abs/2601.09776,ts explanation,ablation,feature,forecast,"
            "yes,yes,,baseline,gap,2026-08-20\n"
        ),
        "uv.lock": "version = 1\n",
        "third_party_manifest/sources.yaml": """schema_version: "1.0.0"
verification_date: "2026-08-20"
sources:
  - source_id: plot
    repository_url: https://github.com/example/plot
    paper_url: https://arxiv.org/abs/2605.06979
    role: baseline
    license_status: UNKNOWN
    license_file: NONE_FOUND
    allowed_action: REFERENCE_ONLY
    default_branch: main
    commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    verification_date: "2026-08-20"
    local_reference_path: null
""",
    }
    for relative_path, content in files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    extra_work_ids = (
        "transport-2608.15645",
        "hyperdas-2503.10894",
        "nonlinear-dilemma-2507.08802",
        "good-apples-2605.02234",
        "rep-divergence-2511.04638",
        "chronos-sae-2603.10071",
        "ts-cbm-2410.06070",
        "forecastcf-2310.08137",
        "foil-2406.09130",
        "cogs-aaai-2026",
    )
    matrix_path = tmp_path / "docs/related_work_matrix.csv"
    with matrix_path.open("a", encoding="utf-8", newline="") as handle:
        for work_id in extra_work_ids:
            handle.write(
                f"{work_id},Title,2026,preprint,https://example.org/{work_id},"
                "problem,intervention,axis,output,no,no,,reference,gap,2026-08-20\n"
            )
    return tmp_path
