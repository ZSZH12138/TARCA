import { useEffect, useState } from "react";

import type { MonitoringApi, RuntimeSnapshot } from "./types";

export function useRuntimeSnapshot(api: MonitoringApi) {
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    void api
      .loadSnapshot()
      .then((next) => {
        if (active) {
          setSnapshot(next);
          setError(null);
        }
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : "监控数据加载失败");
        }
      });
    const unsubscribe = api.subscribe(
      (next) => {
        if (active) {
          setSnapshot(next);
          setError(null);
        }
      },
      (reason) => {
        if (active) setError(reason.message);
      },
    );
    return () => {
      active = false;
      unsubscribe();
    };
  }, [api]);

  return { snapshot, error };
}
