import { formatBytes, formatDecimal, formatEta, formatTimestamp, shortId } from "../format";
import type { JobStatus } from "../types";

const STATE_PRIORITY: Readonly<Record<string, number>> = {
  RUNNING: 0,
  FAILED: 1,
  READY: 2,
  PENDING: 3,
  COMPLETED: 4,
};

function processStatus(job: JobStatus): string {
  if (job.state === "RUNNING") {
    return job.alive ? `PID ${job.pid}` : "未检测到进程";
  }
  if (job.state === "PENDING") return "等待依赖，尚未启动";
  if (job.state === "READY") return "等待资源，尚未启动";
  if (job.state === "COMPLETED") return "任务已完成";
  return "进程已结束";
}

function gpuPlacement(job: JobStatus): string {
  if (job.gpu_ids.length) return job.gpu_ids.join(", ");
  if (job.expected_vram_bytes <= 0) return "CPU";
  return job.state === "PENDING" || job.state === "READY" ? "GPU 待分配" : "GPU 已释放";
}

export function JobTable({ jobs }: { jobs: JobStatus[] }) {
  const running = jobs.filter((job) => job.state === "RUNNING");
  const orderedJobs = jobs
    .map((job, planIndex) => ({ job, planIndex }))
    .sort((left, right) =>
      (STATE_PRIORITY[left.job.state] ?? 5) - (STATE_PRIORITY[right.job.state] ?? 5)
      || left.planIndex - right.planIndex,
    )
    .map(({ job }) => job);
  const activeGpuIds = Array.from(new Set(running.flatMap((job) => job.gpu_ids))).sort();
  const parallelLabel = activeGpuIds.length
    ? `${running.length} 个进程并行运行 · ${activeGpuIds.map((gpu) => `GPU ${gpu}`).join(" / ")}`
    : `${running.length} 个进程运行中`;
  return (
    <section className="panel jobs" aria-labelledby="jobs-title">
      <div className="section-heading compact">
        <div><p className="eyebrow">ISOLATED WORKERS</p><h2 id="jobs-title">运行任务</h2></div>
        <p>{parallelLabel}</p>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>任务</th><th>世界 / 模型</th><th>状态</th><th>GPU</th><th>CPU（实际 / 预留）</th><th>CPU 亲和性</th><th>内存 / 显存</th><th>最近心跳 / 告警</th><th>进度</th><th>当前任务 ETA</th></tr></thead>
          <tbody>
            {orderedJobs.map((job) => (
              <tr key={job.task_id}>
                <td><code title={job.task_id}>{shortId(job.task_id)}</code><small>seed {job.seed ?? "—"}</small></td>
                <td><strong>{job.world_id ?? "—"}</strong><small>{job.model_id ?? job.phase}</small></td>
                <td><span className={`status status-${job.state.toLowerCase()}`}>{job.state}</span><small>{processStatus(job)}</small></td>
                <td>{gpuPlacement(job)}</td>
                <td>{formatDecimal(job.actual_effective_busy_cores)} / {job.expected_cpu_cores}</td>
                <td>{job.cpu_affinity_ids?.length ? job.cpu_affinity_ids.join(", ") : "—"}</td>
                <td>
                  <span>{formatBytes(job.actual_rss_bytes)} / {formatBytes(job.expected_ram_bytes)} RAM</span>
                  <small>{job.expected_vram_bytes > 0 ? `${formatBytes(job.actual_vram_bytes)} / ${formatBytes(job.expected_vram_bytes)} VRAM` : "CPU-only"}</small>
                </td>
                <td><span>{formatTimestamp(job.heartbeat_at_utc)}</span><small>{job.error_category ?? "无告警"}</small></td>
                <td>{job.epoch === null ? "等待" : `Epoch ${job.epoch}`}<small>{job.batch === null ? "" : `Batch ${job.batch}`}</small></td>
                <td>{job.eta_seconds === null ? "待校准" : formatEta(job.eta_seconds, "AVAILABLE")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
