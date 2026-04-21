-- =============================================================================
-- ESG Data Foundry — PostgreSQL Schema
-- Extensions: pgvector (ANN search), pg_trgm (BM25-like trigram search),
--             uuid-ossp, btree_gin
--
-- Self-Check 3.0 patch: valid_from_date + is_superseded on every chunk row.
-- RAG queries MUST filter WHERE is_superseded = FALSE.
-- =============================================================================

-- Extensions -----------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;        -- pgvector ANN
CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- trigram similarity (BM25 proxy)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS btree_gin;     -- GIN index on scalar columns

-- =============================================================================
-- source_documents: tracks ingested PDF files
-- =============================================================================
CREATE TABLE IF NOT EXISTS source_documents (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    filename        TEXT        NOT NULL UNIQUE,
    file_hash       CHAR(64)    NOT NULL,          -- SHA-256 for dedup
    directive_name  TEXT,                          -- e.g. 'CSRD', 'SFDR'
    directive_year  SMALLINT,
    valid_from_date DATE,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    raw_markdown    TEXT                           -- full parsed markdown
);

CREATE INDEX IF NOT EXISTS idx_source_doc_hash ON source_documents (file_hash);

-- =============================================================================
-- directive_chunks: core table — one row per semantic chunk
--
-- Status FSM:  new → in_progress → ready
--                                → unresolvable  (after MAX_RETRIES failures)
--
-- Self-Check 3.0: is_superseded + valid_from_date let Ingestor mark old law.
-- =============================================================================
CREATE TABLE IF NOT EXISTS directive_chunks (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    source_doc_id   UUID        NOT NULL REFERENCES source_documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER     NOT NULL,
    content         TEXT        NOT NULL,          -- plain text (for FTS / BM25)
    content_md      TEXT,                          -- markdown with tables
    -- Vector embedding (text-embedding-3-small = 1536 dims)
    embedding       vector(1536),
    -- Provenance & legal validity
    valid_from_date DATE,
    is_superseded   BOOLEAN     NOT NULL DEFAULT FALSE,
    superseded_by   UUID        REFERENCES directive_chunks(id),
    section_heading TEXT,                          -- nearest H1/H2 heading
    -- ACID processing status (Self-Check: idempotency)
    status          TEXT        NOT NULL DEFAULT 'new'
                        CHECK (status IN ('new','in_progress','ready','unresolvable')),
    retry_count     SMALLINT    NOT NULL DEFAULT 0,
    error_log       TEXT,
    -- Timestamps
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Partial index for fast "fetch next pending chunk" query
CREATE INDEX IF NOT EXISTS idx_chunks_pending
    ON directive_chunks (created_at)
    WHERE status IN ('new', 'in_progress');

-- Index for RAG filter (Self-Check 3.0 — exclude superseded law)
CREATE INDEX IF NOT EXISTS idx_chunks_valid
    ON directive_chunks (is_superseded, valid_from_date DESC)
    WHERE is_superseded = FALSE;

-- IVFFlat ANN index — create only when embeddings exist to avoid low-recall init warning.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM directive_chunks
        WHERE embedding IS NOT NULL
        LIMIT 1
    ) THEN
        EXECUTE '
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON directive_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        ';
    ELSE
        RAISE NOTICE 'Skipping idx_chunks_embedding creation (no embeddings yet).';
    END IF;
END;
$$;

-- GIN trigram index for BM25-like keyword search
CREATE INDEX IF NOT EXISTS idx_chunks_trgm
    ON directive_chunks
    USING gin (content gin_trgm_ops);

-- Full-text search vector (PostgreSQL native FTS as BM25 complement)
ALTER TABLE directive_chunks ADD COLUMN IF NOT EXISTS fts_vector TSVECTOR;

CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON directive_chunks
    USING gin (fts_vector);

-- Trigger: keep fts_vector in sync and update updated_at
CREATE OR REPLACE FUNCTION update_chunk_fts()
RETURNS TRIGGER AS $$
BEGIN
    NEW.fts_vector := to_tsvector('simple', COALESCE(NEW.content, ''));
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_chunk_fts ON directive_chunks;
CREATE TRIGGER trg_chunk_fts
    BEFORE INSERT OR UPDATE OF content
    ON directive_chunks
    FOR EACH ROW EXECUTE FUNCTION update_chunk_fts();

-- =============================================================================
-- generated_samples: output records (one per question-answer pair)
-- =============================================================================
CREATE TABLE IF NOT EXISTS generated_samples (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id        UUID        NOT NULL REFERENCES directive_chunks(id) ON DELETE CASCADE,
    question        TEXT        NOT NULL,
    answer          TEXT        NOT NULL,
    system_prompt   TEXT        NOT NULL,
    is_adversarial  BOOLEAN     NOT NULL DEFAULT FALSE,
    -- Judge metadata
    quality_score   REAL        CHECK (quality_score BETWEEN 0.0 AND 1.0),
    judge_model     TEXT,
    judge_reasoning TEXT,
    -- Extended metadata (perspective / type / difficulty)
    perspective     TEXT,                          -- cfo | prawnik | audytor | cross_doc
    question_type   TEXT,                          -- factual | scope | process | compliance | comparative
    difficulty      TEXT,                          -- easy | medium | hard
    -- DPO pairing
    rejected_answer TEXT,                          -- first-attempt answer for DPO pairs
    conversation_json JSONB,                       -- multi-turn conversation history
    -- Human review workflow
    human_reviewed  BOOLEAN,
    human_flag      TEXT,                          -- auto_approved | auto_rejected | human_approved | human_rejected
    -- Watermarking (Self-Check B2B protection)
    watermark_hash  CHAR(64),
    batch_id        TEXT,
    record_index    INTEGER,   -- position in output file (for watermark locating)
    -- Written to disk?
    written_to_file BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_samples_chunk   ON generated_samples (chunk_id);
CREATE INDEX IF NOT EXISTS idx_samples_batch   ON generated_samples (batch_id);
CREATE INDEX IF NOT EXISTS idx_samples_written ON generated_samples (written_to_file);

-- =============================================================================
-- watermark_registry: maps batch → linguistic signature hash (B2B protection)
-- =============================================================================
CREATE TABLE IF NOT EXISTS watermark_registry (
    id                  UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id            TEXT    NOT NULL UNIQUE,
    client_id           TEXT    NOT NULL,
    watermark_signature TEXT    NOT NULL,  -- human-readable description of technique
    watermark_hash      CHAR(64) NOT NULL, -- SHA-256 of (client_id + batch_id + secret)
    record_indices      INTEGER[],         -- which record positions carry the mark
    total_records       INTEGER,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================================================
-- openai_batch_jobs: tracks async Batch API submissions (Self-Check 2.0)
-- =============================================================================
CREATE TABLE IF NOT EXISTS openai_batch_jobs (
    id              UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_job_id    TEXT    NOT NULL UNIQUE,   -- OpenAI batch job ID
    status          TEXT    NOT NULL DEFAULT 'submitted'
                        CHECK (status IN ('submitted','in_progress','completed','failed')),
    input_file_id   TEXT,
    output_file_id  TEXT,
    sample_ids      UUID[],                    -- generated_samples rows in this batch
    submitted_at    TIMESTAMPTZ DEFAULT NOW(),
    completed_at    TIMESTAMPTZ,
    error_detail    TEXT
);

-- =============================================================================
-- Stored procedure: atomic chunk status transition (ACID idempotency guard)
-- Usage: SELECT claim_chunk_for_processing('<uuid>');
-- Returns TRUE if successfully claimed, FALSE if already taken.
-- =============================================================================
CREATE OR REPLACE FUNCTION claim_chunk_for_processing(p_chunk_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    rows_updated INTEGER;
BEGIN
    UPDATE directive_chunks
    SET    status = 'in_progress', updated_at = NOW()
    WHERE  id = p_chunk_id
      AND  status = 'new'
      AND  retry_count < 3;   -- MAX_RETRIES hard-coded guard

    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    RETURN rows_updated = 1;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- scout_domain_history: rolling log of domains selected per scout run.
-- Gap Scout reads this table on startup to avoid re-scanning the same domains
-- in consecutive runs (persistent across container restarts).
-- =============================================================================
CREATE TABLE IF NOT EXISTS scout_domain_history (
    id          BIGSERIAL   PRIMARY KEY,
    domain_text TEXT        NOT NULL,
    run_id      TEXT        NOT NULL DEFAULT '',
    selected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sdh_selected_at ON scout_domain_history (selected_at DESC);

-- =============================================================================
-- scout_domain_exclusions: domains permanently removed from future scans.
-- Populated when a user clicks "Ingestuj" — the domain is being worked on and
-- MUST NOT appear in future Gap Scout runs.
-- =============================================================================
CREATE TABLE IF NOT EXISTS scout_domain_exclusions (
    topic_id    TEXT        PRIMARY KEY,
    domain_text TEXT        NOT NULL,
    topic_title TEXT        NOT NULL DEFAULT '',
    excluded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sde_domain ON scout_domain_exclusions (domain_text);

-- =============================================================================
-- Stored procedure: mark chunk ready + bump retry on failure (ACID)
-- =============================================================================
CREATE OR REPLACE FUNCTION finalize_chunk(
    p_chunk_id  UUID,
    p_success   BOOLEAN,
    p_error     TEXT DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    IF p_success THEN
        UPDATE directive_chunks
        SET status = 'ready', updated_at = NOW()
        WHERE id = p_chunk_id;
    ELSE
        UPDATE directive_chunks
        SET
            retry_count = retry_count + 1,
            status = CASE
                        WHEN retry_count + 1 >= 3 THEN 'unresolvable'
                        ELSE 'new'
                     END,
            error_log = p_error,
            updated_at = NOW()
        WHERE id = p_chunk_id;
    END IF;
END;
$$ LANGUAGE plpgsql;
