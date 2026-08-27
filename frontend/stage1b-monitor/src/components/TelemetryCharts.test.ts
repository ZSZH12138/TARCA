import { describe, expect, it } from "vitest";

import { telemetrySeriesData } from "./TelemetryCharts";
import { twoGpuSnapshot } from "../test/fixtures";

describe("telemetry chart data", () => {
  it("keeps missing utilization as null instead of fabricating zero", () => {
    const resources = [
      { ...twoGpuSnapshot.resources[0], utilization_percent: null },
      { ...twoGpuSnapshot.resources[1], utilization_percent: 0 },
    ];

    expect(telemetrySeriesData(resources)).toEqual([null, 0]);
  });
});
