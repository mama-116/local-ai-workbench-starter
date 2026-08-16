CREATE TABLE IF NOT EXISTS canonical_memory_event_attributes (
    event_id TEXT NOT NULL REFERENCES canonical_memory_events(id),
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    attribute_key TEXT NOT NULL CHECK(length(trim(attribute_key)) > 0),
    attribute_value TEXT NOT NULL CHECK(length(trim(attribute_value)) > 0),
    source_message_id TEXT NOT NULL,
    PRIMARY KEY(event_id, attribute_key),
    FOREIGN KEY(source_message_id, conversation_id)
        REFERENCES messages(id, conversation_id),
    FOREIGN KEY(event_id, conversation_id)
        REFERENCES canonical_memory_events(id, conversation_id)
);

CREATE INDEX IF NOT EXISTS idx_canonical_memory_attributes_source
ON canonical_memory_event_attributes(conversation_id, source_message_id);

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_attributes_no_update
BEFORE UPDATE ON canonical_memory_event_attributes
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;

CREATE TRIGGER IF NOT EXISTS trg_canonical_memory_attributes_no_delete
BEFORE DELETE ON canonical_memory_event_attributes
BEGIN
    SELECT RAISE(ABORT, 'canonical memory is append-only');
END;
