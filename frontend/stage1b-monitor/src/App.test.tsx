import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "./App";
import { fakeApi, twoGpuSnapshot } from "./test/fixtures";
import type { RuntimeSnapshot } from "./types";

describe("Stage1B runtime dashboard", () => {
  it("uses the runtime API identity and makes parallel work and task ETA explicit", async () => {
    const runtimeAware = {
      ...twoGpuSnapshot,
      runtime: {
        execution_kind: "stage2-v1",
        display_label: "Stage 2 v1",
        access_mode: "READ_ONLY",
      },
      resources: twoGpuSnapshot.resources.map((resource) =>
        resource.kind === "HOST"
          ? {
              ...resource,
              disk_read_bytes_per_second: 1024,
              disk_write_bytes_per_second: 2048,
            }
          : resource,
      ),
      jobs: twoGpuSnapshot.jobs.map((job, index) => ({
        ...job,
        eta_seconds: index === 0 ? 2_400 : 5_400,
        cpu_affinity_ids: index === 0 ? [4, 5, 6, 7] : [8, 9, 10, 11],
      })),
    } as unknown as RuntimeSnapshot;

    render(<App api={fakeApi(runtimeAware)} />);

    expect(await screen.findByText("Stage 2 v1")).toBeVisible();
    expect(screen.getByText(/2 个进程并行运行/)).toBeVisible();
    expect(screen.getByText(/GPU 0.*GPU 1/)).toBeVisible();
    expect(screen.getByText("当前任务 ETA")).toBeVisible();
    expect(screen.getByText("磁盘读取")).toBeVisible();
    expect(screen.getByText("磁盘写入")).toBeVisible();
  });

  it("keeps running work at the top of the task table", async () => {
    const completedFirst = {
      ...twoGpuSnapshot,
      jobs: [
        {
          ...twoGpuSnapshot.jobs[0],
          task_id: "completed-first",
          state: "COMPLETED" as const,
          pid: null,
          alive: false,
          gpu_ids: [],
        },
        {
          ...twoGpuSnapshot.jobs[1],
          task_id: "running-second",
          pid: 901,
          alive: true,
        },
      ],
    };

    render(<App api={fakeApi(completedFirst)} />);

    const rows = await screen.findAllByRole("row");
    expect(rows[1]).toHaveTextContent("PID 901");
    expect(rows[2]).toHaveTextContent("任务已完成");
  });

  it("shows expected and actual resources for both GPUs", async () => {
    render(<App api={fakeApi(twoGpuSnapshot)} />);

    expect(await screen.findByText("GPU 0")).toBeVisible();
    expect(screen.getByText("GPU 1")).toBeVisible();
    expect(screen.getByText("调度预留与实际占用每 2 秒刷新")).toBeVisible();
    expect(screen.getByText("已预留核数")).toBeVisible();
    expect(screen.getByText("CPU（实际 / 预留）")).toBeVisible();
    expect(screen.getAllByText("已预留显存").length).toBeGreaterThan(0);
    expect(screen.getAllByText("实际显存").length).toBeGreaterThan(0);
    expect(screen.getByText("有效忙核")).toBeVisible();
    expect(screen.getByText("预计剩余时间")).toBeVisible();
  });

  it("contains no task mutation controls", async () => {
    render(<App api={fakeApi(twoGpuSnapshot)} />);
    await screen.findByText(twoGpuSnapshot.runtime.display_label);

    expect(screen.queryByRole("button", { name: /重启|停止|删除|修改/ })).toBeNull();
  });

  it("shows calibration state when ETA is unavailable", async () => {
    render(<App api={fakeApi(twoGpuSnapshot)} />);

    expect(await screen.findByText("校准中")).toBeVisible();
  });

  it("labels the conservative preflight ETA without calling it runtime history", async () => {
    const preflightEstimate = {
      ...twoGpuSnapshot,
      run: {
        ...twoGpuSnapshot.run,
        eta_status: "AVAILABLE" as const,
        eta_seconds: 14_400,
        eta_source: "PREFLIGHT_ESTIMATE" as const,
      },
    };

    render(<App api={fakeApi(preflightEstimate)} />);

    expect(await screen.findByText("服务器预检给出的保守估计")).toBeVisible();
  });

  it("distinguishes runtime-history ETA from a failed run", async () => {
    const runtimeEstimate = {
      ...twoGpuSnapshot,
      run: {
        ...twoGpuSnapshot.run,
        eta_status: "AVAILABLE" as const,
        eta_seconds: 3_600,
        eta_source: "RUNTIME_PROGRESS" as const,
      },
    };
    const first = render(<App api={fakeApi(runtimeEstimate)} />);

    expect(await screen.findByText("按已完成同类任务估算")).toBeVisible();
    first.unmount();

    const failed = {
      ...twoGpuSnapshot,
      run: {
        ...twoGpuSnapshot.run,
        status: "FAILED",
        eta_status: "FAILED" as const,
        eta_seconds: null,
        eta_source: "NONE" as const,
      },
    };
    render(<App api={fakeApi(failed)} />);

    expect(await screen.findByText("已有失败任务，需恢复后重算整体 ETA")).toBeVisible();
  });

  it("distinguishes real zero utilization from unavailable telemetry", async () => {
    const liveZero = {
      ...twoGpuSnapshot,
      resources: [{ ...twoGpuSnapshot.resources[0], utilization_percent: 0 }],
    };
    const { unmount } = render(<App api={fakeApi(liveZero)} />);

    expect(await screen.findByText("0%")).toBeVisible();
    expect(screen.getByText("数据正常")).toBeVisible();
    unmount();

    const unavailable = {
      ...twoGpuSnapshot,
      run: { ...twoGpuSnapshot.run, last_sampled_at_utc: null },
      jobs: twoGpuSnapshot.jobs.map((job) => ({
        ...job,
        actual_effective_busy_cores: null,
        actual_rss_bytes: null,
        actual_vram_bytes: null,
      })),
      resources: [{
        ...twoGpuSnapshot.resources[0],
        actual_effective_busy_cores: null,
        actual_memory_bytes: null,
        utilization_percent: null,
        telemetry_status: "UNAVAILABLE" as const,
        sampled_at_utc: null,
      }],
    };
    render(<App api={fakeApi(unavailable)} />);

    expect(await screen.findByText("遥测不可用")).toBeVisible();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("preserves stale measurements while labeling them as expired", async () => {
    const stale = {
      ...twoGpuSnapshot,
      resources: [{
        ...twoGpuSnapshot.resources[1],
        utilization_percent: 93,
        telemetry_status: "STALE" as const,
      }],
    };
    render(<App api={fakeApi(stale)} />);

    expect(await screen.findByText("93%")).toBeVisible();
    expect(screen.getByText("数据过期")).toBeVisible();
    expect(screen.getByText("最后采样")).toBeVisible();
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
      expected_vram_bytes: 0,
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
    expect(screen.getByText("等待依赖，尚未启动")).toBeVisible();
    expect(screen.queryByText("未检测到进程")).not.toBeInTheDocument();
    expect(screen.getByText("等待")).toBeVisible();
    expect(screen.getByText("等待下一批任务")).toBeVisible();
  });

  it("labels an unallocated GPU job by its frozen request instead of calling it CPU", async () => {
    const readyGpu = {
      ...twoGpuSnapshot,
      jobs: [{
        ...twoGpuSnapshot.jobs[0],
        state: "READY",
        pid: null,
        alive: false,
        gpu_ids: [],
        actual_vram_bytes: null,
        epoch: null,
        batch: null,
      }],
    } as RuntimeSnapshot;

    render(<App api={fakeApi(readyGpu)} />);

    expect(await screen.findByText("GPU 待分配")).toBeVisible();
    expect(screen.getByText("— / 20.0 GiB VRAM")).toBeVisible();
  });
});
