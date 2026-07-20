PRAGMA foreign_keys = ON;

CREATE TABLE turn_batches (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    branch_id TEXT NOT NULL REFERENCES branches(id),
    source_message_id TEXT NOT NULL REFERENCES messages(id),
    response_message_id TEXT NOT NULL UNIQUE REFERENCES messages(id),
    run_id TEXT NOT NULL UNIQUE REFERENCES runs(id),
    model TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('story', 'round_table', 'spotlight')),
    formal_character_ids_json TEXT NOT NULL,
    guest_ids_json TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('completed', 'partial')),
    repair_state TEXT NOT NULL CHECK (repair_state IN ('not_needed', 'succeeded', 'failed')),
    error_code TEXT,
    created_at TEXT NOT NULL,
    CHECK (
        (state = 'completed' AND error_code IS NULL)
        OR (state = 'partial' AND length(trim(error_code)) > 0)
    )
);

CREATE TABLE turn_segments (
    id TEXT PRIMARY KEY,
    turn_batch_id TEXT NOT NULL REFERENCES turn_batches(id),
    position INTEGER NOT NULL CHECK (position >= 0),
    speaker_kind TEXT NOT NULL CHECK (
        speaker_kind IN ('character', 'guest', 'narrator', 'unresolved')
    ),
    speaker_id TEXT,
    display_name TEXT NOT NULL CHECK (length(trim(display_name)) > 0),
    content TEXT NOT NULL CHECK (length(trim(content)) > 0),
    UNIQUE(turn_batch_id, position),
    CHECK (
        (speaker_kind IN ('character', 'guest') AND speaker_id IS NOT NULL)
        OR (speaker_kind IN ('narrator', 'unresolved') AND speaker_id IS NULL)
    )
);

CREATE INDEX idx_turn_batches_conversation
ON turn_batches(conversation_id, created_at);

CREATE INDEX idx_turn_segments_batch
ON turn_segments(turn_batch_id, position);
