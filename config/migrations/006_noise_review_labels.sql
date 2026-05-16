ALTER TABLE crops ADD COLUMN noise_review_label TEXT;
ALTER TABLE crops ADD COLUMN noise_review_note TEXT;
ALTER TABLE crops ADD COLUMN noise_reviewed_at TEXT;
ALTER TABLE crops ADD COLUMN noise_review_score_col TEXT;
PRAGMA user_version = 6;
