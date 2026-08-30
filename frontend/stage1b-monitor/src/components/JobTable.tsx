import { formatBytes, formatDecimal, formatTimestamp, shortId } from "../format";
import type { JobStatus } from "../types";

function processStatus(job: JobStatus): string {
  if (job.state === "RUNNING") {
    return job.alive ? `PID ${job.pid}` : "未检测到进程";
  }
  if (job.state === "PENDING") return "等待依赖，尚未启动";
  if (job.state === "READY") return "等待资源，尚未启动";
  if (job.state === "COMPLETED") return "任务已完成";
  return "进程已结束";
}

export function JobTable({ jobs }: { jobs: JobStatus[] }) {
  return (
    <section className="panel jobs" aria-labelledby="jobs-title">
      <div className="section-heading compact">
        <div><p className="eyebrow">ISOLATED WORKERS</p><h2 id="jobs-title">运行任务</h2></div>
        <p>{jobs.filter((job) => job.state === "RUNNING").length} 个进程运行中</p>
      </div>
      <div className="table-wrap">
        <table>
          <thead><tr><th>任务</th><th>世界 / 模型</th><th>状态</th><th>GPU</th><th>CPU（实际 / 预留）</th><th>内存 / 显存</th><th>最近心跳 / 告警</th><th>进度</th></tr></thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.task_id}>
                <td><code title={job.task_id}>{shortId(job.task_id)}</code><small>seed {job.seed ?? "—"}</small></td>
                <td><strong>{job.world_id ?? "—"}</strong><small>{job.model_id ?? job.phase}</small></td>
                <td><span className={`status status-${job.state.toLowerCase()}`}>{job.state}</span><small>{processStatus(job)}</small></td>
                <td>{job.gpu_ids.length ? job.gpu_ids.join(", ") : "CPU"}</td>
                <td>{formatDecimal(job.actual_effective_busy_cores)} / {job.expected_cpu_cores}</td>
                <td><span>{formatBytes(job.actual_rss_bytes)}</span><small>{job.gpu_ids.length ? `${formatBytes(job.actual_vram_bytes)} VRAM` : "—"}</small></td>
                <td><span>{formatTimestamp(job.heartbeat_at_utc)}</span><small>{job.error_category ?? "无告警"}</small></td>
                <td>{job.epoch === null ? "等待" : `Epoch ${job.epoch}`}<small>{job.batch === null ? "" : `Batch ${job.batch}`}</small></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
