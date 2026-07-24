ALTER TABLE model_role_settings RENAME TO model_role_settings_legacy;

CREATE TABLE model_role_settings (
    role TEXT PRIMARY KEY CHECK (
        role IN ('embedding', 'memory_extraction')
    ),
    desired_profile_id TEXT REFERENCES embedding_profiles(id),
    active_profile_id TEXT REFERENCES embedding_profiles(id),
    connection_id TEXT,
    provider_name TEXT,
    endpoint_fingerprint TEXT,
    model_name TEXT,
    model_digest TEXT,
    updated_at TEXT NOT NULL,
    CHECK (
        (
            role = 'embedding'
            AND connection_id IS NULL
            AND provider_name IS NULL
            AND endpoint_fingerprint IS NULL
            AND model_name IS NULL
            AND model_digest IS NULL
        )
        OR
        (
            role = 'memory_extraction'
            AND desired_profile_id IS NULL
            AND active_profile_id IS NULL
            AND connection_id IS NOT NULL
            AND provider_name IS NOT NULL
            AND endpoint_fingerprint IS NOT NULL
            AND model_name IS NOT NULL
            AND model_digest IS NOT NULL
        )
    )
);

INSERT INTO model_role_settings(
    role, desired_profile_id, active_profile_id, updated_at
)
SELECT role, desired_profile_id, active_profile_id, updated_at
FROM model_role_settings_legacy;

DROP TABLE model_role_settings_legacy;
