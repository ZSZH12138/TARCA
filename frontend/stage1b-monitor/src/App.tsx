import { browserMonitoringApi } from "./api";
import { AlertPanel } from "./components/AlertPanel";
import { JobTable } from "./components/JobTable";
import { ResourceGrid } from "./components/ResourceGrid";
import { RunSummary } from "./components/RunSummary";
import { TelemetryCharts } from "./components/TelemetryCharts";
import type { MonitoringApi } from "./types";
import { useRuntimeSnapshot } from "./useRuntimeSnapshot";

export function App({ api = browserMonitoringApi }: { api?: MonitoringApi }) {
  const { snapshot, error } = useRuntimeSnapshot(api);
  const connectingLabel = import.meta.env.VITE_TARCA_CONNECTING_LABEL ?? "正在连接 Stage1B 运行时…";

  if (snapshot === null) {
    return <main className="loading"><span /><p>{error ?? connectingLabel}</p></main>;
  }
  return (
    <main className="dashboard-shell">
      <header><div className="brand-mark">T</div><div><strong>TARCA</strong><span>运行监督台</span></div><p>只读模式</p></header>
      {error ? <div className="connection-alert" role="alert">{error}</div> : null}
      <RunSummary summary={snapshot.run} />
      <ResourceGrid resources={snapshot.resources} />
      <div className="insight-grid"><TelemetryCharts resources={snapshot.resources} /><AlertPanel alerts={snapshot.alerts} /></div>
      <JobTable jobs={snapshot.jobs} />
      <footer>该页面只展示运行状态，不提供任务修改操作。</footer>
    </main>
  );
}
