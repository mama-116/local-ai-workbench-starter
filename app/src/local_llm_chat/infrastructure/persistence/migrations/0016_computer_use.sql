CREATE TABLE IF NOT EXISTS computer_use_runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    objective TEXT NOT NULL,
    observation_id TEXT NOT NULL,
    plan_hash TEXT NOT NULL CHECK(length(plan_hash) = 64),
    max_actions INTEGER NOT NULL CHECK(max_actions BETWEEN 1 AND 5),
    max_duration_seconds REAL NOT NULL CHECK(
        max_duration_seconds > 0 AND max_duration_seconds <= 60
    ),
    approval_timeout_seconds REAL NOT NULL CHECK(
        approval_timeout_seconds > 0 AND approval_timeout_seconds <= 30
    ),
    planned_action_count INTEGER NOT NULL CHECK(
        planned_action_count BETWEEN 1 AND 5
        AND planned_action_count <= max_actions
    ),
    state TEXT NOT NULL CHECK(state IN (
        'pending', 'awaiting_approval', 'running', 'completed',
        'failed', 'cancelled', 'denied'
    )),
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_single_running_computer_use_run
ON computer_use_runs((1))
WHERE state = 'running';

CREATE TABLE IF NOT EXISTS computer_plan_approvals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES computer_use_runs(id),
    plan_hash TEXT NOT NULL CHECK(length(plan_hash) = 64),
    approved_at TEXT NOT NULL,
    consumed_at TEXT
);

CREATE TABLE IF NOT EXISTS computer_actions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES computer_use_runs(id),
    ordinal INTEGER NOT NULL CHECK(ordinal > 0),
    request_id TEXT NOT NULL,
    action_type TEXT NOT NULL CHECK(action_type IN (
        'launch_allowed_app', 'click_uia_element', 'type_plain_text'
    )),
    target_profile_id TEXT NOT NULL,
    input_text TEXT,
    state TEXT NOT NULL CHECK(state IN (
        'proposed', 'approved', 'running', 'completed',
        'failed', 'cancelled', 'denied'
    )),
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    UNIQUE(run_id, ordinal),
    UNIQUE(run_id, request_id)
);

CREATE INDEX IF NOT EXISTS idx_computer_actions_run
ON computer_actions(run_id, ordinal);
