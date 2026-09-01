from __future__ import annotations

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    graph_id TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_plan_nodes (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    task_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    node_json TEXT NOT NULL,
    PRIMARY KEY (run_id, task_id),
    UNIQUE (run_id, ordinal)
);
CREATE TABLE IF NOT EXISTS job_nodes (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    phase TEXT NOT NULL,
    executor_key TEXT NOT NULL,
    output_artifact_type TEXT NOT NULL,
    scientific_identity_json TEXT NOT NULL,
    resource_request_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS task_specs (
    task_id TEXT PRIMARY KEY REFERENCES job_nodes(task_id),
    spec_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dependencies (
    task_id TEXT NOT NULL REFERENCES task_specs(task_id),
    dependency_task_id TEXT NOT NULL REFERENCES task_specs(task_id),
    input_ordinal INTEGER NOT NULL CHECK (input_ordinal >= 0),
    PRIMARY KEY (task_id, dependency_task_id),
    UNIQUE (task_id, input_ordinal)
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES task_specs(task_id),
    attempt_number INTEGER NOT NULL CHECK (attempt_number >= 1),
    state TEXT NOT NULL CHECK (state IN ('READY','RUNNING','COMPLETED','FAILED','STALLED')),
    worker_id TEXT,
    allocation_json TEXT,
    pid INTEGER,
    process_started_at_utc TEXT,
    heartbeat_at_utc TEXT,
    error_category TEXT,
    artifact_json TEXT,
    packing_level INTEGER NOT NULL DEFAULT 1 CHECK (packing_level >= 1),
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    UNIQUE (task_id, attempt_number)
);
CREATE TABLE IF NOT EXISTS progress_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
    recorded_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS resource_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    attempt_id TEXT REFERENCES attempts(attempt_id),
    sampled_at_utc TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    attempt_id TEXT REFERENCES attempts(attempt_id),
    created_at_utc TEXT NOT NULL,
    category TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_events (
    recovery_id TEXT NOT NULL,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    source_attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
    new_attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
    reason TEXT NOT NULL,
    spec_sha256 TEXT NOT NULL,
    checkpoint_sha256 TEXT NOT NULL,
    code_bundle_sha256 TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    PRIMARY KEY (recovery_id, source_attempt_id),
    UNIQUE (new_attempt_id)
);
CREATE INDEX IF NOT EXISTS idx_attempts_state ON attempts(state, created_at_utc);
CREATE INDEX IF NOT EXISTS idx_progress_attempt ON progress_events(attempt_id, event_id);
CREATE INDEX IF NOT EXISTS idx_recovery_events_run ON recovery_events(run_id, recovery_id);
"""
