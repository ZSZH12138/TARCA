import { useEffect, useRef } from "react";

import type { ResourceStatus } from "../types";

export function TelemetryCharts({ resources }: { resources: ResourceStatus[] }) {
  const chartRoot = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const root = chartRoot.current;
    if (root === null) return;
    let disposed = false;
    let disposeChart = () => undefined;
    void import("echarts").then((echarts) => {
      if (disposed) return;
      const chart = echarts.init(root, undefined, { renderer: "canvas" });
      chart.setOption({
        backgroundColor: "transparent",
        grid: { left: 35, right: 16, top: 18, bottom: 32 },
        xAxis: {
          type: "category",
          data: resources.map((item) => item.label),
          axisLabel: { color: "#8b99a7" },
          axisLine: { lineStyle: { color: "#273540" } },
        },
        yAxis: {
          type: "value",
          max: 100,
          axisLabel: { color: "#8b99a7", formatter: "{value}%" },
          splitLine: { lineStyle: { color: "#1d2a33" } },
        },
        series: [{
          type: "bar",
          data: resources.map((item) => item.utilization_percent),
          itemStyle: { color: "#40e0b6", borderRadius: [5, 5, 0, 0] },
          barMaxWidth: 42,
        }],
      });
      const observer = new ResizeObserver(() => chart.resize());
      observer.observe(root);
      disposeChart = () => {
        observer.disconnect();
        chart.dispose();
      };
    });
    return () => {
      disposed = true;
      disposeChart();
    };
  }, [resources]);

  return (
    <section className="panel chart-panel" aria-labelledby="chart-title">
      <div className="section-heading compact"><div><p className="eyebrow">TELEMETRY</p><h2 id="chart-title">实时利用率</h2></div></div>
      <div className="chart" ref={chartRoot} role="img" aria-label="主机与显卡利用率图表" />
    </section>
  );
}
