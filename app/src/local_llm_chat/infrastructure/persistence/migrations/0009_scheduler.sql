CREATE TABLE IF NOT EXISTS scheduled_jobs (
    id TEXT PRIMARY KEY,
    handler_name TEXT NOT NULL,
    interval_seconds INTEGER NOT NULL CHECK (interval_seconds > 0),
    first_due_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_runs (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES scheduled_jobs(id),
    scheduled_for TEXT NOT NULL,
    attempt INTEGER NOT NULL CHECK (attempt > 0),
    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'completed', 'failed')),
    retry_of_run_id TEXT REFERENCES job_runs(id),
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    UNIQUE(job_id, scheduled_for, attempt)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_job_runs_one_running_per_job
ON job_runs(job_id) WHERE state = 'running';

CREATE INDEX IF NOT EXISTS idx_job_runs_job_schedule
ON job_runs(job_id, scheduled_for, attempt);
