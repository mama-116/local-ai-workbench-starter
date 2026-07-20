ALTER TABLE conversations
ADD COLUMN auto_translate INTEGER NOT NULL DEFAULT 0
CHECK (auto_translate IN (0, 1));

ALTER TABLE branches
ADD COLUMN hidden_at TEXT;
