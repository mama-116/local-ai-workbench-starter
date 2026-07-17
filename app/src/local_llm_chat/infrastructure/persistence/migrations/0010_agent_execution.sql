CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    objective TEXT NOT NULL,
    allowed_tools_json TEXT NOT NULL,
    max_cost_units INTEGER NOT NULL CHECK(max_cost_units BETWEEN 1 AND 5),
    max_steps INTEGER NOT NULL CHECK(max_steps BETWEEN 1 AND 5),
    max_duration_seconds REAL NOT NULL CHECK(
        max_duration_seconds > 0 AND max_duration_seconds <= 60
    ),
    state TEXT NOT NULL CHECK(state IN (
        'pending', 'running', 'completed', 'failed', 'cancelled', 'denied'
    )),
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_single_running_agent_run
ON agent_runs((1))
WHERE state = 'running';

CREATE TABLE IF NOT EXISTS agent_steps (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agent_runs(id),
    ordinal INTEGER NOT NULL CHECK(ordinal > 0),
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    action_hash TEXT NOT NULL,
    data_classification TEXT CHECK(data_classification IN (
        'private', 'local_operational', 'search_query', 'public_result'
    )),
    effect TEXT CHECK(effect IN (
        'read', 'reversible_write', 'external_post', 'external_delete', 'billing'
    )),
    destination TEXT CHECK(destination IN ('local', 'lan', 'remote', 'unknown')),
    cost_class TEXT CHECK(cost_class IN (
        'no_charge', 'free_tier', 'paid', 'unknown'
    )),
    cost_units INTEGER CHECK(cost_units BETWEEN 1 AND 5),
    state TEXT NOT NULL CHECK(state IN (
        'proposed', 'running', 'completed', 'failed', 'denied',
        'restored', 'restore_failed'
    )),
    result_size_bytes INTEGER CHECK(result_size_bytes >= 0),
    result_sha256 TEXT,
    restore_token TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    UNIQUE(run_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_agent_steps_run
ON agent_steps(run_id, ordinal);
