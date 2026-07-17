CREATE TABLE run_rag_usage (
    run_id TEXT PRIMARY KEY REFERENCES runs(id),
    selected_document_count INTEGER NOT NULL CHECK (selected_document_count > 0)
);
