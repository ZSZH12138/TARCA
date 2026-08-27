export interface RunSummary {
  run_id: string;
  graph_id: string;
  status: string;
  phase: string | null;
  total_tasks: number;
  completed_tasks: number;
  running_tasks: number;
  failed_tasks: number;
  pending_tasks: number;
  progress_fraction: number;
  eta_seconds: number | null;
  eta_status: "CALIBRATING" | "AVAILABLE" | "COMPLETE" | "FAILED";
  created_at_utc: string;
  last_sampled_at_utc: string | null;
}

export interface JobStatus {
  task_id: string;
  phase: string;
  world_id: string | null;
  model_id: string | null;
  seed: number | null;
  state: string;
  pid: number | null;
  alive: boolean;
  gpu_ids: number[];
  expected_cpu_cores: number;
  actual_effective_busy_cores: number | null;
  expected_ram_bytes: number;
  actual_rss_bytes: number | null;
  expected_vram_bytes: number;
  actual_vram_bytes: number | null;
  epoch: number | null;
  batch: number | null;
  heartbeat_at_utc: string | null;
  retry_count: number;
  eta_seconds: number | null;
  error_category: string | null;
}

export interface ResourceStatus {
  resource_id: string;
  label: string;
  kind: "HOST" | "GPU";
  expected_cpu_cores: number;
  actual_effective_busy_cores: number | null;
  expected_memory_bytes: number;
  actual_memory_bytes: number | null;
  utilization_percent: number | null;
  temperature_celsius: number | null;
  power_watts: number | null;
  active_processes: number;
  sampled_at_utc: string | null;
  telemetry_status: "LIVE" | "STALE" | "UNAVAILABLE";
}

export interface RuntimeAlert {
  alert_id: number;
  task_id: string | null;
  category: string;
  message: string;
  created_at_utc: string;
}

export interface RuntimeSnapshot {
  run: RunSummary;
  jobs: JobStatus[];
  resources: ResourceStatus[];
  alerts: RuntimeAlert[];
}

export interface MonitoringApi {
  loadSnapshot(): Promise<RuntimeSnapshot>;
  subscribe(
    onSnapshot: (snapshot: RuntimeSnapshot) => void,
    onError: (error: Error) => void,
  ): () => void;
}
