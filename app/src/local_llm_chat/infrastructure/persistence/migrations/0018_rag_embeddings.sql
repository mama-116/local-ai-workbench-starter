CREATE TABLE IF NOT EXISTS embedding_profiles (
    id TEXT PRIMARY KEY,
    connection_id TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    endpoint_fingerprint TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    vector_dimensions INTEGER NOT NULL CHECK (vector_dimensions > 0),
    min_similarity REAL NOT NULL CHECK (
        min_similarity >= -1.0 AND min_similarity <= 1.0
    ),
    state TEXT NOT NULL CHECK (state IN ('pending', 'building', 'ready', 'failed')),
    total_chunks INTEGER NOT NULL DEFAULT 0 CHECK (total_chunks >= 0),
    embedded_chunks INTEGER NOT NULL DEFAULT 0 CHECK (embedded_chunks >= 0),
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(
        connection_id, endpoint_fingerprint, model_name,
        model_digest, vector_dimensions
    )
);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    profile_id TEXT NOT NULL REFERENCES embedding_profiles(id) ON DELETE CASCADE,
    content_hash TEXT NOT NULL,
    vector_blob BLOB NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(chunk_id, profile_id)
);

CREATE TABLE IF NOT EXISTS model_role_settings (
    role TEXT PRIMARY KEY CHECK (role = 'embedding'),
    desired_profile_id TEXT REFERENCES embedding_profiles(id),
    active_profile_id TEXT REFERENCES embedding_profiles(id),
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_profile
ON chunk_embeddings(profile_id, chunk_id);
