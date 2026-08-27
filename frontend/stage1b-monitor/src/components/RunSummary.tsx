import { formatEta, formatTimestamp, shortId } from "../format";
import type { RunSummary as RunSummaryType } from "../types";

export function RunSummary({ summary }: { summary: RunSummaryType }) {
  const percent = Math.round(summary.progress_fraction * 100);
  return (
    <section className="summary panel" aria-labelledby="run-title">
      <div className="summary-copy">
        <p className="eyebrow">QUALIFICATION RUNTIME</p>
        <h1 id="run-title">Stage1B v2</h1>
        <p className="run-id" title={summary.run_id}>{shortId(summary.run_id)}</p>
      </div>
      <div className="progress-block">
        <div className="progress-label">
          <span>任务进度</span>
          <strong>{summary.completed_tasks} / {summary.total_tasks}</strong>
        </div>
        <div className="progress-track" aria-label={`任务进度 ${percent}%`}>
          <span style={{ width: `${percent}%` }} />
        </div>
        <p>{summary.phase ?? "等待下一批任务"}</p>
      </div>
      <div className="eta-block">
        <span>预计剩余时间</span>
        <strong>{formatEta(summary.eta_seconds, summary.eta_status)}</strong>
        <small className={`status status-${summary.status.toLowerCase()}`}>{summary.status}</small>
        <small className="sample-time"><span>最后采样</span><time>{formatTimestamp(summary.last_sampled_at_utc)}</time></small>
      </div>
    </section>
  );
}
