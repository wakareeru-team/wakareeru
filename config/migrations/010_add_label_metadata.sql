-- Canonical and sole source for the generated dataset/l10n_metadata.json artifact.
-- images.wiki_title/operator_en/operator_jp remain in place as legacy per-image
-- metadata for pipeline and review compatibility; localized export must not read them.
CREATE TABLE IF NOT EXISTS label_metadata (
    label_ja          TEXT PRIMARY KEY,
    label_en          TEXT NOT NULL,
    label_zh          TEXT NOT NULL,
    operator_ja_json  TEXT NOT NULL DEFAULT '[]',
    operator_en_json  TEXT NOT NULL DEFAULT '[]',
    operator_zh_json  TEXT NOT NULL DEFAULT '[]',
    wiki_title_ja     TEXT NOT NULL DEFAULT '',
    note              TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

PRAGMA user_version = 10;
