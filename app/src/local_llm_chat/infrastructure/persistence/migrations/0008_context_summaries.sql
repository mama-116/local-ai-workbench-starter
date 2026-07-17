CREATE TABLE IF NOT EXISTS context_summaries (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    branch_id TEXT NOT NULL REFERENCES branches(id),
    source_message_ids_json TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    settings_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    content TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'completed', 'failed')),
    error_code TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_context_summaries_reuse
ON context_summaries(
    conversation_id,
    branch_id,
    source_hash,
    settings_hash,
    prompt_version,
    state,
    created_at DESC
);
