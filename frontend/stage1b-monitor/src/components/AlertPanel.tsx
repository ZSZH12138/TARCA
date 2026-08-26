import type { RuntimeAlert } from "../types";

export function AlertPanel({ alerts }: { alerts: RuntimeAlert[] }) {
  return (
    <section className="panel alerts" aria-labelledby="alerts-title">
      <div className="section-heading compact"><div><p className="eyebrow">RUNTIME SIGNALS</p><h2 id="alerts-title">运行提醒</h2></div><span className="alert-count">{alerts.length}</span></div>
      {alerts.length === 0 ? <p className="empty">当前没有运行提醒</p> : (
        <ul>{alerts.map((alert) => <li key={alert.alert_id}><span>{alert.category}</span><p>{alert.message}</p><time>{new Date(alert.created_at_utc).toLocaleTimeString("zh-CN")}</time></li>)}</ul>
      )}
    </section>
  );
}
