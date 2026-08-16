PRAGMA foreign_keys = ON;

ALTER TABLE character_versions
ADD COLUMN relationship_attachment_pace TEXT NOT NULL DEFAULT 'standard'
CHECK(relationship_attachment_pace IN ('slow', 'standard', 'quick'));

ALTER TABLE character_versions
ADD COLUMN relationship_expressiveness TEXT NOT NULL DEFAULT 'balanced'
CHECK(relationship_expressiveness IN ('reserved', 'balanced', 'expressive'));

ALTER TABLE character_versions
ADD COLUMN relationship_priority TEXT NOT NULL DEFAULT 'shared_experience'
CHECK(relationship_priority IN (
    'words', 'commitments', 'boundaries', 'shared_experience'
));

ALTER TABLE character_versions
ADD COLUMN relationship_conflict_response TEXT NOT NULL DEFAULT 'direct'
CHECK(relationship_conflict_response IN ('withdraw', 'direct', 'repair_seeking'));

ALTER TABLE character_versions
ADD COLUMN relationship_recovery_pace TEXT NOT NULL DEFAULT 'standard'
CHECK(relationship_recovery_pace IN ('slow', 'standard', 'quick'));

ALTER TABLE relationship_events
ADD COLUMN affinity_delta INTEGER CHECK(affinity_delta BETWEEN -5 AND 5);

ALTER TABLE relationship_events
ADD COLUMN trust_delta INTEGER CHECK(trust_delta BETWEEN -5 AND 5);

ALTER TABLE relationship_events
ADD COLUMN tension_delta INTEGER CHECK(tension_delta BETWEEN -10 AND 10);

CREATE TRIGGER relationship_event_resolved_delta_all_or_none_insert
BEFORE INSERT ON relationship_events
WHEN (
    (NEW.affinity_delta IS NULL) +
    (NEW.trust_delta IS NULL) +
    (NEW.tension_delta IS NULL)
) NOT IN (0, 3)
BEGIN
    SELECT RAISE(ABORT, 'relationship deltas must be all null or all set');
END;

CREATE TRIGGER relationship_event_resolved_delta_all_or_none_update
BEFORE UPDATE OF affinity_delta, trust_delta, tension_delta ON relationship_events
WHEN (
    (NEW.affinity_delta IS NULL) +
    (NEW.trust_delta IS NULL) +
    (NEW.tension_delta IS NULL)
) NOT IN (0, 3)
BEGIN
    SELECT RAISE(ABORT, 'relationship deltas must be all null or all set');
END;
