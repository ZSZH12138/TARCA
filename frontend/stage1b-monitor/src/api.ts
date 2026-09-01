import type { MonitoringApi, RuntimeIdentity, RuntimeSnapshot } from "./types";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { method: "GET", credentials: "same-origin" });
  if (!response.ok) {
    throw new Error(`监控接口返回 ${response.status}`);
  }
  return (await response.json()) as T;
}

function websocketUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/api/v1/stream`;
}

export function createBrowserMonitoringApi(): MonitoringApi {
  let runtimePromise: Promise<RuntimeIdentity> | null = null;
  const loadRuntime = (refresh = false): Promise<RuntimeIdentity> => {
    if (refresh || runtimePromise === null) {
      runtimePromise = getJson<RuntimeIdentity>("/api/v1/runtime");
    }
    return runtimePromise;
  };

  const loadSnapshot = async (): Promise<RuntimeSnapshot> => {
    const [runtime, run, jobs, resources, alerts] = await Promise.all([
      loadRuntime(),
      getJson<RuntimeSnapshot["run"]>("/api/v1/run"),
      getJson<RuntimeSnapshot["jobs"]>("/api/v1/jobs"),
      getJson<RuntimeSnapshot["resources"]>("/api/v1/resources"),
      getJson<RuntimeSnapshot["alerts"]>("/api/v1/alerts"),
    ]);
    return { runtime, run, jobs, resources, alerts };
  };

  return {
    loadSnapshot,
    subscribe(onSnapshot, onError) {
      let stopped = false;
      let reconnectAttempt = 0;
      let socket: WebSocket | null = null;
      let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

      const connect = () => {
        if (stopped) return;
        socket = new WebSocket(websocketUrl());
        socket.onopen = () => {
          reconnectAttempt = 0;
          void loadRuntime(true).catch(() => undefined);
        };
        socket.onmessage = (event) => {
          try {
            const value = JSON.parse(String(event.data)) as
              | RuntimeSnapshot
              | Omit<RuntimeSnapshot, "runtime">;
            if ("runtime" in value) {
              onSnapshot(value);
            } else {
              void loadRuntime()
                .then((runtime) => onSnapshot({ ...value, runtime }))
                .catch((error: unknown) => {
                  onError(error instanceof Error ? error : new Error("运行身份加载失败"));
                });
            }
          } catch {
            onError(new Error("监控数据格式无效"));
          }
        };
        socket.onerror = () => socket?.close();
        socket.onclose = () => {
          if (stopped) return;
          runtimePromise = null;
          void loadSnapshot().then(onSnapshot).catch((error: unknown) => {
            onError(error instanceof Error ? error : new Error("监控连接暂时中断"));
          });
          const delay = Math.min(30_000, 1_000 * 2 ** reconnectAttempt);
          reconnectAttempt += 1;
          reconnectTimer = setTimeout(connect, delay);
        };
      };

      connect();
      return () => {
        stopped = true;
        if (reconnectTimer !== null) clearTimeout(reconnectTimer);
        socket?.close();
      };
    },
  };
}

export const browserMonitoringApi = createBrowserMonitoringApi();
