ALTER TABLE crops ADD COLUMN manual_corrected_label TEXT;
ALTER TABLE crops ADD COLUMN manual_corrected_at TEXT;
PRAGMA user_version = 9;
