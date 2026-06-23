CREATE TABLE IF NOT EXISTS category_tree_checkpoints (
    series           TEXT NOT NULL,
    root_category    TEXT NOT NULL,
    category         TEXT NOT NULL,
    remaining_depth  INTEGER NOT NULL,
    completed_at     TEXT NOT NULL,
    PRIMARY KEY (series, root_category, category)
);

PRAGMA user_version = 12;
