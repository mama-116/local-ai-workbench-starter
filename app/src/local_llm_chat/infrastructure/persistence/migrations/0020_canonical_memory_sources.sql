CREATE TABLE IF NOT EXISTS canonical_memory_event_sources (
    event_id TEXT NOT NULL REFERENCES canonical_memory_events(id),
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    source_message_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    PRIMARY KEY(event_id, source_message_id),
    UNIQUE(event_id, ordinal),
    FOREIGN KEY(source_message_id, conversation_id)
        REFERENCES messages(id, conversation_id),
    FOREIGN KEY(event_id, conversation_id)
        REFERENCES canonical_memory_events(id, conversation_id)
);

INSERT OR IGNORE INTO canonical_memory_event_sources(
    event_id, conversation_id, source_message_id, ordinal
)
SELECT id, conversation_id, source_message_id, 0
FROM canonical_memory_events;

CREATE INDEX IF NOT EXISTS idx_canonical_memory_sources_message
ON canonical_memory_event_sources(conversation_id, source_message_id);

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_sources_no_update
BEFORE UPDATE ON canonical_memory_event_sources
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_sources_no_delete
BEFORE DELETE ON canonical_memory_event_sources
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;
