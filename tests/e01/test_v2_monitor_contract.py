from __future__ import annotations

from pathlib import Path

from tarca.monitoring.repository import SAFE_JOB_COLUMNS
from tarca.monitoring.schemas import RunSummaryView

ROOT = Path(__file__).resolve().parents[2]


def test_monitor_contract_contains_required_operational_fields_only() -> None:
    run_fields = set(RunSummaryView.model_fields)
    required_jobs = {
        "expected_cpu_cores",
        "actual_effective_busy_cores",
        "expected_ram_bytes",
        "actual_rss_bytes",
        "expected_vram_bytes",
        "actual_vram_bytes",
        "heartbeat_at_utc",
        "eta_seconds",
    }

    assert {
        "total_tasks",
        "completed_tasks",
        "running_tasks",
        "pending_tasks",
        "failed_tasks",
        "eta_seconds",
        "eta_status",
        "last_sampled_at_utc",
        "last_checkpoint_at_utc",
    } <= run_fields
    assert required_jobs <= set(SAFE_JOB_COLUMNS)
    forbidden = {"crps", "gate", "win_rate", "scientific_result", "multiplier_bias"}
    assert forbidden.isdisjoint(run_fields)
    assert forbidden.isdisjoint(SAFE_JOB_COLUMNS)


def test_frontend_renders_cpu_ram_gpu_vram_eta_checkpoint_and_read_only_notice() -> None:
    resource = (ROOT / "frontend/stage1b-monitor/src/components/ResourceGrid.tsx").read_text(
        encoding="utf-8"
    )
    summary = (ROOT / "frontend/stage1b-monitor/src/components/RunSummary.tsx").read_text(
        encoding="utf-8"
    )
    jobs = (ROOT / "frontend/stage1b-monitor/src/components/JobTable.tsx").read_text(
        encoding="utf-8"
    )
    app = (ROOT / "frontend/stage1b-monitor/src/App.tsx").read_text(encoding="utf-8")

    for label in ("已预留核数", "有效忙核", "已预留内存", "实际内存"):
        assert label in resource
    for label in ("已预留显存", "实际显存"):
        assert label in resource
    assert "预计剩余时间" in summary
    assert "等待首批完成任务校准" in summary
    assert "last_checkpoint_at_utc" in summary
    assert "最近检查点" in summary
    assert "heartbeat_at_utc" in jobs
    assert "最近心跳" in jobs
    assert "error_category" in jobs
    assert "只读模式" in app
    assert "不提供任务修改操作" in app
