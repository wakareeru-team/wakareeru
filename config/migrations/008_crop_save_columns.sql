ALTER TABLE crops ADD COLUMN saved INTEGER NOT NULL DEFAULT 0; -- 0: not saved, 1: saved to disk
ALTER TABLE crops ADD COLUMN crop_path TEXT; -- relative path to saved crop image, e.g. "crops/123.jpg"
PRAGMA user_version = 8;