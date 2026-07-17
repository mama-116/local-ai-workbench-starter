ALTER TABLE runs ADD COLUMN generation_duration_ns INTEGER;
ALTER TABLE runs ADD COLUMN response_duration_ms INTEGER;

CREATE TABLE telemetry_samples (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    metric_name TEXT NOT NULL,
    value REAL,
    unit TEXT NOT NULL,
    source TEXT NOT NULL,
    unavailable_reason TEXT,
    captured_at TEXT NOT NULL,
    CHECK (value IS NOT NULL OR unavailable_reason IS NOT NULL)
);

CREATE INDEX idx_telemetry_run ON telemetry_samples(run_id, captured_at);
