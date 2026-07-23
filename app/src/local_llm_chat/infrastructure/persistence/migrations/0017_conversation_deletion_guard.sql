CREATE TABLE conversation_deletion_guards (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    authorized_at TEXT NOT NULL
);

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

DROP TRIGGER trg_canonical_memory_events_no_delete;
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

DROP TRIGGER trg_canonical_memory_knowledge_no_delete;
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

DROP TRIGGER trg_canonical_memory_decisions_no_delete;
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
