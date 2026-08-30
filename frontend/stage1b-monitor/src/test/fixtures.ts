import type { MonitoringApi, RuntimeSnapshot } from "../types";

export const twoGpuSnapshot: RuntimeSnapshot = {
  run: { run_id: "run-a", graph_id: "graph-a", status: "ACTIVE", phase: "NEURAL_TRAIN", total_tasks: 74, completed_tasks: 30, running_tasks: 2, failed_tasks: 0, pending_tasks: 42, progress_fraction: 30 / 74, eta_seconds: null, eta_status: "CALIBRATING", created_at_utc: "2026-08-26T12:00:00Z", last_sampled_at_utc: "2026-08-26T12:01:00Z", last_checkpoint_at_utc: "2026-08-26T12:00:55Z" },
  jobs: [0, 1].map((gpu) => ({ task_id: `stage1b-neural-train-${gpu}-abcdef123456`, phase: "NEURAL_TRAIN", world_id: gpu === 0 ? "lorenz96_f10_v2" : "lorenz96_twoscale_v2", model_id: gpu === 0 ? "itransformer_reference" : "patchtst_reference", seed: 104729, state: "RUNNING", pid: 300 + gpu, alive: true, gpu_ids: [gpu], expected_cpu_cores: 4, actual_effective_busy_cores: 3.7, expected_ram_bytes: 32 * 1024 ** 3, actual_rss_bytes: 11 * 1024 ** 3, expected_vram_bytes: 20 * 1024 ** 3, actual_vram_bytes: (17 + gpu) * 1024 ** 3, epoch: 12, batch: 44, heartbeat_at_utc: "2026-08-26T12:01:00Z", retry_count: 0, eta_seconds: null, error_category: null })),
  resources: [
    { resource_id: "host", label: "主机", kind: "HOST", expected_cpu_cores: 8, actual_effective_busy_cores: 19.2, expected_memory_bytes: 64 * 1024 ** 3, actual_memory_bytes: 87 * 1024 ** 3, utilization_percent: 76, temperature_celsius: null, power_watts: null, active_processes: 2, sampled_at_utc: "2026-08-26T12:01:00Z", telemetry_status: "LIVE" },
    ...[0, 1].map((gpu) => ({ resource_id: `gpu-${gpu}`, label: `GPU ${gpu}`, kind: "GPU" as const, expected_cpu_cores: 4, actual_effective_busy_cores: null, expected_memory_bytes: 20 * 1024 ** 3, actual_memory_bytes: (17 + gpu) * 1024 ** 3, utilization_percent: 93 - gpu * 4, temperature_celsius: 71 + gpu, power_watts: 402 - gpu * 8, active_processes: 1, sampled_at_utc: "2026-08-26T12:01:00Z", telemetry_status: "LIVE" as const })),
  ],
  alerts: [{ alert_id: 1, task_id: null, category: "ETA_CALIBRATION", message: "完成首个稳定窗口后生成预计剩余时间", created_at_utc: "2026-08-26T12:00:30Z" }],
};

export interface FakeMonitoringApi extends MonitoringApi {
  emitError(error: Error): void;
  emitSnapshot(snapshot: RuntimeSnapshot): void;
}

export function fakeApi(snapshot: RuntimeSnapshot): FakeMonitoringApi {
  let errorHandler: (error: Error) => void = () => undefined;
  let snapshotHandler: (snapshot: RuntimeSnapshot) => void = () => undefined;
  return {
    loadSnapshot: async () => snapshot,
    subscribe(onSnapshot, onError) {
      snapshotHandler = onSnapshot;
      errorHandler = onError;
      return () => undefined;
    },
    emitError(error) { errorHandler(error); },
    emitSnapshot(next) { snapshotHandler(next); },
  };
}
