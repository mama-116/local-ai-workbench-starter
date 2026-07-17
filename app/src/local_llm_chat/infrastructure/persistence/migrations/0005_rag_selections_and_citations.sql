CREATE TABLE conversation_documents (
    conversation_id TEXT NOT NULL REFERENCES conversations(id),
    document_id TEXT NOT NULL REFERENCES documents(id),
    selected_at TEXT NOT NULL,
    PRIMARY KEY(conversation_id, document_id)
);

CREATE TABLE run_citations (
    run_id TEXT NOT NULL REFERENCES runs(id),
    chunk_id TEXT NOT NULL REFERENCES chunks(id),
    rank INTEGER NOT NULL CHECK (rank >= 0),
    score REAL NOT NULL CHECK (score > 0),
    PRIMARY KEY(run_id, chunk_id),
    UNIQUE(run_id, rank)
);

CREATE INDEX idx_conversation_documents_conversation
ON conversation_documents(conversation_id, selected_at);

CREATE INDEX idx_run_citations_run ON run_citations(run_id, rank);
