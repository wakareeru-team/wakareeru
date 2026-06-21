ALTER TABLE images
ADD COLUMN llm_metadata_processed INTEGER NOT NULL DEFAULT 0;

-- Existing rows are the protected baseline. Stage 06 must not replay its CSV
-- checkpoint over them after this migration; newly crawled rows keep DEFAULT 0.
UPDATE images SET llm_metadata_processed = 1;

PRAGMA user_version = 11;
