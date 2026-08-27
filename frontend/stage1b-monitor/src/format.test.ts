import { describe, expect, it } from "vitest";

import { formatBytes, formatEta, shortId } from "./format";

describe("runtime display formatting", () => {
  it("formats byte and identifier boundaries", () => {
    expect(formatBytes(0)).toBe("0 GiB");
    expect(formatBytes(1.5 * 1024 ** 3)).toBe("1.5 GiB");
    expect(formatBytes(null)).toBe("—");
    expect(shortId("short-id")).toBe("short-id");
    expect(shortId("stage1b-very-long-scientific-task-id")).toContain("…");
  });

  it("formats calibration, completion, and positive ETA", () => {
    expect(formatEta(null, "CALIBRATING")).toBe("校准中");
    expect(formatEta(0, "COMPLETE")).toBe("已完成");
    expect(formatEta(7_260, "AVAILABLE")).toBe("2 小时 1 分钟");
  });
});
