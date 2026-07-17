CREATE TABLE IF NOT EXISTS message_translations (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    source_hash TEXT NOT NULL,
    target_language TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'completed', 'failed')),
    error_code TEXT,
    reused_from_id TEXT REFERENCES message_translations(id),
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_translations_message
ON message_translations(message_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_translations_cache
ON message_translations(
    source_hash,
    target_language,
    provider,
    model,
    state,
    completed_at DESC
);
