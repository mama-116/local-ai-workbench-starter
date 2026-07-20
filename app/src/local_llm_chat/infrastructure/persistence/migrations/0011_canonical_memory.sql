CREATE UNIQUE INDEX IF NOT EXISTS uq_branches_id_conversation
ON branches(id, conversation_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_messages_id_conversation
ON messages(id, conversation_id);

CREATE TABLE IF NOT EXISTS canonical_memory_events (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    branch_id TEXT NOT NULL,
    subject_id TEXT NOT NULL CHECK(length(trim(subject_id)) > 0),
    kind TEXT NOT NULL CHECK(kind IN ('preference', 'safety_constraint', 'goal')),
    slot TEXT NOT NULL CHECK(length(trim(slot)) > 0),
    value TEXT NOT NULL CHECK(length(trim(value)) > 0),
    cardinality TEXT NOT NULL CHECK(cardinality IN ('single', 'multiple')),
    approval TEXT NOT NULL CHECK(approval IN (
        'auto_saved', 'confirmed', 'pending_confirmation'
    )),
    knowledge_count INTEGER NOT NULL CHECK(knowledge_count >= 0),
    source_message_id TEXT NOT NULL,
    supersedes_event_id TEXT,
    effective_at TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    CHECK(supersedes_event_id IS NULL OR supersedes_event_id <> id),
    UNIQUE(id, conversation_id),
    FOREIGN KEY(branch_id, conversation_id) REFERENCES branches(id, conversation_id),
    FOREIGN KEY(source_message_id, conversation_id) REFERENCES messages(id, conversation_id),
    FOREIGN KEY(supersedes_event_id, conversation_id)
        REFERENCES canonical_memory_events(id, conversation_id)
);

CREATE TABLE IF NOT EXISTS canonical_memory_event_knowledge (
    event_id TEXT NOT NULL REFERENCES canonical_memory_events(id),
    character_id TEXT NOT NULL REFERENCES characters(id),
    PRIMARY KEY(event_id, character_id)
);

CREATE TABLE IF NOT EXISTS canonical_memory_decisions (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    target_event_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN ('confirmed', 'rejected', 'undone')),
    source_message_id TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    FOREIGN KEY(target_event_id, conversation_id)
        REFERENCES canonical_memory_events(id, conversation_id),
    FOREIGN KEY(source_message_id, conversation_id)
        REFERENCES messages(id, conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_canonical_memory_events_conversation
ON canonical_memory_events(conversation_id, recorded_at, id);

CREATE INDEX IF NOT EXISTS idx_canonical_memory_decisions_target
ON canonical_memory_decisions(target_event_id, sequence);

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_events_no_update
BEFORE UPDATE ON canonical_memory_events
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_events_no_delete
BEFORE DELETE ON canonical_memory_events
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_knowledge_no_update
BEFORE UPDATE ON canonical_memory_event_knowledge
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_knowledge_capacity
BEFORE INSERT ON canonical_memory_event_knowledge
WHEN (
    SELECT COUNT(*) FROM canonical_memory_event_knowledge
    WHERE event_id = NEW.event_id
) >= (
    SELECT knowledge_count FROM canonical_memory_events
    WHERE id = NEW.event_id
)
BEGIN
    SELECT RAISE(ABORT, 'canonical memory knowledge scope is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_knowledge_no_delete
BEFORE DELETE ON canonical_memory_event_knowledge
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_decisions_no_update
BEFORE UPDATE ON canonical_memory_decisions
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_decisions_no_delete
BEFORE DELETE ON canonical_memory_decisions
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_decision_transition
BEFORE INSERT ON canonical_memory_decisions
BEGIN
    SELECT CASE
        WHEN NEW.state = 'confirmed' AND COALESCE(
            (SELECT state FROM canonical_memory_decisions
             WHERE target_event_id = NEW.target_event_id
             ORDER BY sequence DESC LIMIT 1),
            (SELECT approval FROM canonical_memory_events
             WHERE id = NEW.target_event_id)
        ) <> 'pending_confirmation'
        THEN RAISE(ABORT, 'invalid canonical memory decision transition')
        WHEN NEW.state = 'rejected' AND COALESCE(
            (SELECT state FROM canonical_memory_decisions
             WHERE target_event_id = NEW.target_event_id
             ORDER BY sequence DESC LIMIT 1),
            (SELECT approval FROM canonical_memory_events
             WHERE id = NEW.target_event_id)
        ) <> 'pending_confirmation'
        THEN RAISE(ABORT, 'invalid canonical memory decision transition')
        WHEN NEW.state = 'undone' AND COALESCE(
            (SELECT state FROM canonical_memory_decisions
             WHERE target_event_id = NEW.target_event_id
             ORDER BY sequence DESC LIMIT 1),
            (SELECT approval FROM canonical_memory_events
             WHERE id = NEW.target_event_id)
        ) NOT IN ('auto_saved', 'confirmed')
        THEN RAISE(ABORT, 'invalid canonical memory decision transition')
    END;
END;
