import { afterEach, describe, expect, it, vi } from "vitest";

import { createBrowserMonitoringApi } from "./api";
import { twoGpuSnapshot } from "./test/fixtures";
import type { RuntimeSnapshot } from "./types";

describe("browser monitoring API", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("loads the five read-only REST views", async () => {
    const runtimeIdentity = {
      execution_kind: "stage2-v1",
      display_label: "Stage 2 v1",
      access_mode: "READ_ONLY",
    };
    const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
      const path = String(input);
      const value = path.endsWith("/runtime")
        ? runtimeIdentity
        : path.endsWith("/run")
        ? twoGpuSnapshot.run
        : path.endsWith("/jobs")
          ? twoGpuSnapshot.jobs
          : path.endsWith("/resources")
            ? twoGpuSnapshot.resources
            : twoGpuSnapshot.alerts;
      return new Response(JSON.stringify(value), { status: 200 });
    });
    vi.stubGlobal("fetch", fetchMock);

    const snapshot = await createBrowserMonitoringApi().loadSnapshot();

    expect("runtime" in (snapshot as object)).toBe(true);
    expect(snapshot.run.run_id).toBe("run-a");
    expect(fetchMock).toHaveBeenCalledTimes(5);
    expect(fetchMock.mock.calls.every((call) => call[1]?.method === "GET")).toBe(true);
  });

  it("rejects a failed REST response", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("offline", { status: 503 })));

    await expect(createBrowserMonitoringApi().loadSnapshot()).rejects.toThrow("503");
  });

  it("streams safe snapshots and reports malformed websocket messages", async () => {
    class SocketStub {
      static instances: SocketStub[] = [];
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      close = vi.fn();
      constructor(public url: string) { SocketStub.instances.push(this); }
    }
    vi.stubGlobal("WebSocket", SocketStub);
    const snapshots: RuntimeSnapshot[] = [];
    const errors: Error[] = [];

    const unsubscribe = createBrowserMonitoringApi().subscribe(
      (snapshot) => snapshots.push(snapshot),
      (error) => errors.push(error),
    );
    const socket = SocketStub.instances[0];
    socket.onopen?.();
    socket.onmessage?.({ data: JSON.stringify(twoGpuSnapshot) } as MessageEvent);
    socket.onmessage?.({ data: "not-json" } as MessageEvent);
    socket.onerror?.();
    unsubscribe();

    expect(socket.url).toBe("ws://localhost:3000/api/v1/stream");
    expect(snapshots).toEqual([twoGpuSnapshot]);
    expect(errors[0].message).toBe("监控数据格式无效");
    expect(socket.close).toHaveBeenCalled();
  });

  it("falls back to REST and caps websocket reconnect delay", async () => {
    vi.useFakeTimers();
    class SocketStub {
      static instances: SocketStub[] = [];
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      constructor(_url: string) { SocketStub.instances.push(this); }
      close() { this.onclose?.(); }
    }
    vi.stubGlobal("WebSocket", SocketStub);
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const path = String(input);
      const value = path.endsWith("/runtime") ? twoGpuSnapshot.runtime
        : path.endsWith("/run") ? twoGpuSnapshot.run
        : path.endsWith("/jobs") ? twoGpuSnapshot.jobs
          : path.endsWith("/resources") ? twoGpuSnapshot.resources
            : twoGpuSnapshot.alerts;
      return new Response(JSON.stringify(value), { status: 200 });
    }));
    const onSnapshot = vi.fn();
    const onError = vi.fn();
    const unsubscribe = createBrowserMonitoringApi().subscribe(onSnapshot, onError);

    SocketStub.instances[0].onclose?.();
    await vi.waitFor(() => expect(onSnapshot).toHaveBeenCalledWith(twoGpuSnapshot));
    await vi.advanceTimersByTimeAsync(1_000);

    expect(SocketStub.instances).toHaveLength(2);
    expect(onError).not.toHaveBeenCalled();
    unsubscribe();
  });

  it("normalizes non-Error fallback failures", async () => {
    vi.useFakeTimers();
    class SocketStub {
      static instance: SocketStub;
      onopen: (() => void) | null = null;
      onmessage: ((event: MessageEvent) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      constructor(_url: string) { SocketStub.instance = this; }
      close() {}
    }
    vi.stubGlobal("WebSocket", SocketStub);
    vi.stubGlobal("fetch", vi.fn(async () => Promise.reject("offline")));
    const onError = vi.fn();
    const unsubscribe = createBrowserMonitoringApi().subscribe(vi.fn(), onError);

    SocketStub.instance.onclose?.();
    await vi.waitFor(() => expect(onError).toHaveBeenCalled());

    expect(onError.mock.calls[0][0].message).toBe("监控连接暂时中断");
    unsubscribe();
  });
});
