import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "./App";
import { fakeApi, twoGpuSnapshot } from "./test/fixtures";

describe("Stage1B runtime dashboard", () => {
  it("shows expected and actual resources for both GPUs", async () => {
    render(<App api={fakeApi(twoGpuSnapshot)} />);

    expect(await screen.findByText("GPU 0")).toBeVisible();
    expect(screen.getByText("GPU 1")).toBeVisible();
    expect(screen.getAllByText("期望显存").length).toBeGreaterThan(0);
    expect(screen.getAllByText("实际显存").length).toBeGreaterThan(0);
    expect(screen.getByText("有效忙核")).toBeVisible();
    expect(screen.getByText("预计剩余时间")).toBeVisible();
  });

  it("contains no task mutation controls", async () => {
    render(<App api={fakeApi(twoGpuSnapshot)} />);
    await screen.findByText("Stage1B v2");

    expect(screen.queryByRole("button", { name: /重启|停止|删除|修改/ })).toBeNull();
  });

  it("shows calibration state when ETA is unavailable", async () => {
    render(<App api={fakeApi(twoGpuSnapshot)} />);

    expect(await screen.findByText("校准中")).toBeVisible();
  });

  it("renders connection errors without discarding the last safe snapshot", async () => {
    const api = fakeApi(twoGpuSnapshot);
    render(<App api={api} />);
    await screen.findByText("GPU 0");

    api.emitError(new Error("监控连接暂时中断"));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("监控连接暂时中断"));
    expect(screen.getByText("GPU 0")).toBeVisible();
  });

  it("shows a loading error before any snapshot is available", async () => {
    const api = {
      loadSnapshot: async () => { throw new Error("运行数据库不可用"); },
      subscribe: () => () => undefined,
    };
    render(<App api={api} />);

    expect(await screen.findByText("运行数据库不可用")).toBeVisible();
  });

  it("renders completed ETA, empty alerts, and critical resource pressure", async () => {
    const snapshot = {
      ...twoGpuSnapshot,
      run: { ...twoGpuSnapshot.run, eta_status: "COMPLETE" as const, eta_seconds: 0 },
      alerts: [],
      resources: twoGpuSnapshot.resources.map((resource) => ({
        ...resource,
        utilization_percent: resource.kind === "GPU" ? 96 : 80,
      })),
    };
    render(<App api={fakeApi(snapshot)} />);

    expect(await screen.findByText("已完成")).toBeVisible();
    expect(screen.getByText("当前没有运行提醒")).toBeVisible();
  });

  it("renders CPU waiting jobs and a healthy resource update from the stream", async () => {
    const api = fakeApi(twoGpuSnapshot);
    render(<App api={api} />);
    await screen.findByText("GPU 0");
    const cpuJob = {
      ...twoGpuSnapshot.jobs[0],
      task_id: "cpu-task",
      world_id: null,
      model_id: null,
      state: "PENDING",
      pid: null,
      alive: false,
      gpu_ids: [],
      actual_effective_busy_cores: 0,
      actual_rss_bytes: 0,
      actual_vram_bytes: 0,
      epoch: null,
      batch: null,
      seed: null,
    };
    api.emitSnapshot({
      ...twoGpuSnapshot,
      run: { ...twoGpuSnapshot.run, phase: null },
      jobs: [cpuJob],
      resources: [{ ...twoGpuSnapshot.resources[0], utilization_percent: 12 }],
    });

    expect(await screen.findByText("CPU")).toBeVisible();
    expect(screen.getByText("未检测到进程")).toBeVisible();
    expect(screen.getByText("等待")).toBeVisible();
    expect(screen.getByText("等待下一批任务")).toBeVisible();
  });
});
