from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"
REQUIRED_DOCS = (
    DOCS_DIR / "assumption_ledger.md",
    DOCS_DIR / "literature_audit_log.md",
    DOCS_DIR / "novelty_claims.md",
    DOCS_DIR / "preregistration_v0.md",
    DOCS_DIR / "related_work_matrix.csv",
    DOCS_DIR / "stage0_scope.md",
    DOCS_DIR / "terminology.md",
)
MANDATORY_CAUSAL_BOUNDARY_SENTENCES = (
    (
        "\u0054\u0041\u0052\u0043\u0041 \u9996\u7bc7\u7814\u7a76\u53ea\u80fd\u5bf9"
        "\u6a21\u578b\u5185\u90e8\u8ba1\u7b97\u673a\u5236\u63d0\u51fa"
        "\u56e0\u679c\u9648\u8ff0\u3002"
    ),
    (
        "\u6a21\u578b\u5185\u90e8\u5e72\u9884\u4e00\u81f4\u6027\u4e0d\u80fd"
        "\u81ea\u52a8\u63a8\u51fa\u771f\u5b9e\u91d1\u878d\u5e02\u573a\u4e2d"
        "\u7684\u56e0\u679c\u5173\u7cfb\u3002"
    ),
)


def test_required_stage0_documents_exist_and_are_non_empty() -> None:
    for path in REQUIRED_DOCS:
        assert path.exists(), f"Missing required Stage 0 document: {path}"
        text = path.read_text(encoding="utf-8")
        assert text.strip(), f"Required Stage 0 document is empty: {path}"


def test_terminology_contains_mandatory_causal_boundary_sentences() -> None:
    terminology = (DOCS_DIR / "terminology.md").read_text(encoding="utf-8")

    for sentence in MANDATORY_CAUSAL_BOUNDARY_SENTENCES:
        assert sentence in terminology, (
            f"terminology.md is missing mandatory causal-boundary sentence: {sentence}"
        )
