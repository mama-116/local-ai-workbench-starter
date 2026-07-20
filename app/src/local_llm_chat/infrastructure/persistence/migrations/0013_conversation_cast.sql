PRAGMA foreign_keys = ON;

CREATE TABLE conversation_cast_members (
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    character_id TEXT NOT NULL REFERENCES characters(id),
    character_version_id TEXT NOT NULL REFERENCES character_versions(id),
    position INTEGER NOT NULL CHECK (position BETWEEN 0 AND 4),
    added_at TEXT NOT NULL,
    PRIMARY KEY(conversation_id, character_id),
    UNIQUE(conversation_id, position)
);

INSERT INTO conversation_cast_members(
    conversation_id, character_id, character_version_id, position, added_at
)
SELECT c.id, cv.character_id, c.character_version_id, 0, c.created_at
FROM conversations c
JOIN character_versions cv ON cv.id = c.character_version_id;

CREATE INDEX idx_conversation_cast_order
ON conversation_cast_members(conversation_id, position);
