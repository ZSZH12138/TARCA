"""Audit contract for the Stage 1 Gate 0 reopening record."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LITERATURE_LOG = PROJECT_ROOT / "docs" / "literature_audit_log.md"
NOVELTY_CLAIMS = PROJECT_ROOT / "docs" / "novelty_claims.md"


def test_stage1_gate0_reopening_records_current_primary_sources() -> None:
    literature = LITERATURE_LOG.read_text(encoding="utf-8")
    novelty = NOVELTY_CLAIMS.read_text(encoding="utf-8")
    required_sources = (
        "https://arxiv.org/abs/2605.06979",
        "https://arxiv.org/abs/2510.04842",
        "https://arxiv.org/abs/2510.15821",
        "https://arxiv.org/abs/2506.03128",
        "https://arxiv.org/abs/2607.01204",
        "https://arxiv.org/abs/2606.18367",
    )

    assert "Stage 1 Gate 0 重开记录" in literature
    assert "2026-07-27" in literature
    assert all(source in literature for source in required_sources)
    assert "不是新颖性证明" in literature
    assert "核查日期：`2026-07-27`" in novelty  # noqa: RUF001
    assert "Chronos-2" in novelty
    assert "TiRex-2" in novelty
