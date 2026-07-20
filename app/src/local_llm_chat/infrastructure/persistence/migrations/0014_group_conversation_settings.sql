PRAGMA foreign_keys = ON;

CREATE TABLE conversation_group_settings (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id),
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    mode TEXT NOT NULL CHECK (mode IN ('story', 'round_table', 'spotlight')),
    spotlight_character_id TEXT REFERENCES characters(id),
    updated_at TEXT NOT NULL,
    CHECK (
        (enabled = 0 AND mode = 'story' AND spotlight_character_id IS NULL)
        OR (enabled = 1 AND mode IN ('story', 'round_table')
            AND spotlight_character_id IS NULL)
        OR (enabled = 1 AND mode = 'spotlight'
            AND spotlight_character_id IS NOT NULL)
    )
);

INSERT INTO conversation_group_settings(
    conversation_id, enabled, mode, spotlight_character_id, updated_at
)
SELECT id, 0, 'story', NULL, updated_at FROM conversations;

ALTER TABLE turn_batches
ADD COLUMN spotlight_character_id TEXT REFERENCES characters(id);
