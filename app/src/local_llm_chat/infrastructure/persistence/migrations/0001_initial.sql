PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS character_versions (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(id),
    version INTEGER NOT NULL CHECK (version > 0),
    system_prompt TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(character_id, version)
);

CREATE TABLE IF NOT EXISTS model_profiles (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(provider, model_name)
);

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    active_branch_id TEXT,
    character_version_id TEXT NOT NULL REFERENCES character_versions(id),
    model_profile_id TEXT NOT NULL REFERENCES model_profiles(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    parent_message_id TEXT REFERENCES messages(id),
    source_message_id TEXT REFERENCES messages(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'streaming', 'completed', 'cancelled', 'failed')),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    CHECK (parent_message_id IS NULL OR parent_message_id <> id),
    CHECK (source_message_id IS NULL OR source_message_id <> id)
);

CREATE TABLE IF NOT EXISTS branches (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    parent_branch_id TEXT REFERENCES branches(id),
    forked_from_message_id TEXT REFERENCES messages(id),
    head_message_id TEXT REFERENCES messages(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    request_message_id TEXT NOT NULL REFERENCES messages(id),
    response_message_id TEXT NOT NULL UNIQUE REFERENCES messages(id),
    character_version_id TEXT NOT NULL REFERENCES character_versions(id),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('pending', 'running', 'completed', 'cancelled', 'failed')),
    started_at TEXT,
    completed_at TEXT,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    total_duration_ns INTEGER,
    error_code TEXT
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_events (
    id TEXT PRIMARY KEY,
    level TEXT NOT NULL,
    event_type TEXT NOT NULL,
    run_id TEXT REFERENCES runs(id),
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(archived_at, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_message_id);
CREATE INDEX IF NOT EXISTS idx_branches_conversation ON branches(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_created ON app_events(created_at DESC);
