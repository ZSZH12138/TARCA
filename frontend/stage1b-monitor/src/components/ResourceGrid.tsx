import { formatBytes } from "../format";
import type { ResourceStatus } from "../types";

function meterTone(value: number): string {
  if (value >= 90) return "critical";
  if (value >= 75) return "warm";
  return "healthy";
}

export function ResourceGrid({ resources }: { resources: ResourceStatus[] }) {
  return (
    <section aria-labelledby="resource-title">
      <div className="section-heading">
        <div><p className="eyebrow">LIVE CAPACITY</p><h2 id="resource-title">硬件资源</h2></div>
        <p>期望分配与实际占用每 2 秒刷新</p>
      </div>
      <div className="resource-grid">
        {resources.map((resource) => (
          <article className="resource-card panel" key={resource.resource_id}>
            <div className="resource-topline">
              <h3>{resource.label}</h3>
              <span>{resource.utilization_percent.toFixed(0)}%</span>
            </div>
            <div className="meter"><span className={meterTone(resource.utilization_percent)} style={{ width: `${resource.utilization_percent}%` }} /></div>
            {resource.kind === "HOST" ? (
              <dl>
                <div><dt>期望核数</dt><dd>{resource.expected_cpu_cores}</dd></div>
                <div><dt>有效忙核</dt><dd>{resource.actual_effective_busy_cores.toFixed(1)}</dd></div>
                <div><dt>期望内存</dt><dd>{formatBytes(resource.expected_memory_bytes)}</dd></div>
                <div><dt>实际内存</dt><dd>{formatBytes(resource.actual_memory_bytes)}</dd></div>
              </dl>
            ) : (
              <dl>
                <div><dt>期望显存</dt><dd>{formatBytes(resource.expected_memory_bytes)}</dd></div>
                <div><dt>实际显存</dt><dd>{formatBytes(resource.actual_memory_bytes)}</dd></div>
                <div><dt>温度</dt><dd>{resource.temperature_celsius?.toFixed(0) ?? "—"}°C</dd></div>
                <div><dt>功率</dt><dd>{resource.power_watts?.toFixed(0) ?? "—"} W</dd></div>
              </dl>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}
