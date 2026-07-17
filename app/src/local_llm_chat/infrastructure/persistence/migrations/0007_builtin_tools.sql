CREATE TABLE IF NOT EXISTS conversation_tool_folder_grants (
    conversation_id TEXT PRIMARY KEY,
    root_path TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    revoked_at TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    run_id TEXT,
    provider TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    input_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'completed', 'failed', 'denied')),
    result_content TEXT,
    result_item_count INTEGER,
    result_size_bytes INTEGER,
    result_sha256 TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation
ON tool_calls(conversation_id, created_at DESC);
