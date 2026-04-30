CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    source_path TEXT        NOT NULL,
    chunk_index INT         NOT NULL,
    content     TEXT        NOT NULL,
    metadata    JSONB       NOT NULL DEFAULT '{}',
    embedding   VECTOR(1024),
    indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS abuse_log (
    id            UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    hashed_ip     TEXT        NOT NULL,
    attempt_count INT         NOT NULL DEFAULT 1,
    prompt_text   TEXT,
    abuse_type    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS abuse_log_ip_time_idx
    ON abuse_log (hashed_ip, created_at DESC);
