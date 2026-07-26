CREATE TABLE explicit_memory_events (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    branch_id TEXT NOT NULL,
    source_message_id TEXT NOT NULL,
    character_id TEXT NOT NULL REFERENCES characters(id),
    value TEXT NOT NULL CHECK(
        length(trim(value)) > 0 AND length(value) <= 200
    ),
    recorded_at TEXT NOT NULL,
    UNIQUE(id, conversation_id),
    FOREIGN KEY(branch_id, conversation_id)
        REFERENCES branches(id, conversation_id),
    FOREIGN KEY(source_message_id, conversation_id)
        REFERENCES messages(id, conversation_id)
);

CREATE TABLE explicit_memory_decisions (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    target_event_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK(state = 'undone'),
    recorded_at TEXT NOT NULL,
    FOREIGN KEY(target_event_id, conversation_id)
        REFERENCES explicit_memory_events(id, conversation_id)
);

CREATE INDEX idx_explicit_memory_events_conversation
ON explicit_memory_events(conversation_id, recorded_at, id);

CREATE TRIGGER trg_explicit_memory_events_no_update
BEFORE UPDATE ON explicit_memory_events
BEGIN
    SELECT RAISE(ABORT, 'explicit memory is append-only');
END;

CREATE TRIGGER trg_explicit_memory_events_no_delete
BEFORE DELETE ON explicit_memory_events
WHEN NOT EXISTS (
    SELECT 1
    FROM conversation_deletion_guards
    WHERE conversation_id = OLD.conversation_id
)
BEGIN
    SELECT RAISE(ABORT, 'explicit memory is append-only');
END;

CREATE TRIGGER trg_explicit_memory_decisions_no_update
BEFORE UPDATE ON explicit_memory_decisions
BEGIN
    SELECT RAISE(ABORT, 'explicit memory is append-only');
END;

CREATE TRIGGER trg_explicit_memory_decisions_no_delete
BEFORE DELETE ON explicit_memory_decisions
WHEN NOT EXISTS (
    SELECT 1
    FROM conversation_deletion_guards
    WHERE conversation_id = OLD.conversation_id
)
BEGIN
    SELECT RAISE(ABORT, 'explicit memory is append-only');
END;
