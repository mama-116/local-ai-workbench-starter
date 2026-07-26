-- Compatibility: the pre-merge explicit-memory branch used version 17.
-- Recreate the official version-17 deletion guard when that occupied version
-- caused the canonical migration to be skipped in a development database.
CREATE TABLE IF NOT EXISTS conversation_deletion_guards (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    authorized_at TEXT NOT NULL
);

DROP TRIGGER IF EXISTS trg_conversation_deletion_guard_archived_only;
CREATE TRIGGER trg_conversation_deletion_guard_archived_only
BEFORE INSERT ON conversation_deletion_guards
WHEN NOT EXISTS (
    SELECT 1
    FROM conversations
    WHERE id = NEW.conversation_id
      AND archived_at IS NOT NULL
)
BEGIN
    SELECT RAISE(ABORT, 'only archived conversations can be permanently deleted');
END;

DROP TRIGGER IF EXISTS trg_canonical_memory_events_no_delete;
CREATE TRIGGER trg_canonical_memory_events_no_delete
BEFORE DELETE ON canonical_memory_events
WHEN NOT EXISTS (
    SELECT 1
    FROM conversation_deletion_guards
    WHERE conversation_id = OLD.conversation_id
)
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;

DROP TRIGGER IF EXISTS trg_canonical_memory_knowledge_no_delete;
CREATE TRIGGER trg_canonical_memory_knowledge_no_delete
BEFORE DELETE ON canonical_memory_event_knowledge
WHEN NOT EXISTS (
    SELECT 1
    FROM canonical_memory_events event
    JOIN conversation_deletion_guards guard
      ON guard.conversation_id = event.conversation_id
    WHERE event.id = OLD.event_id
)
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;

DROP TRIGGER IF EXISTS trg_canonical_memory_decisions_no_delete;
CREATE TRIGGER trg_canonical_memory_decisions_no_delete
BEFORE DELETE ON canonical_memory_decisions
WHEN NOT EXISTS (
    SELECT 1
    FROM conversation_deletion_guards
    WHERE conversation_id = OLD.conversation_id
)
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;

CREATE TABLE IF NOT EXISTS explicit_memory_events (
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

CREATE TABLE IF NOT EXISTS explicit_memory_decisions (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    target_event_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK(state = 'undone'),
    recorded_at TEXT NOT NULL,
    FOREIGN KEY(target_event_id, conversation_id)
        REFERENCES explicit_memory_events(id, conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_explicit_memory_events_conversation
ON explicit_memory_events(conversation_id, recorded_at, id);

DROP TRIGGER IF EXISTS trg_explicit_memory_events_no_update;
CREATE TRIGGER trg_explicit_memory_events_no_update
BEFORE UPDATE ON explicit_memory_events
BEGIN
    SELECT RAISE(ABORT, 'explicit memory is append-only');
END;

DROP TRIGGER IF EXISTS trg_explicit_memory_events_no_delete;
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

DROP TRIGGER IF EXISTS trg_explicit_memory_decisions_no_update;
CREATE TRIGGER trg_explicit_memory_decisions_no_update
BEFORE UPDATE ON explicit_memory_decisions
BEGIN
    SELECT RAISE(ABORT, 'explicit memory is append-only');
END;

DROP TRIGGER IF EXISTS trg_explicit_memory_decisions_no_delete;
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
