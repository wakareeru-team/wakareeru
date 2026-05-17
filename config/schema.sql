PRAGMA foreign_keys = ON;

-- Commons category tree nodes crawled from Wikimedia Commons.
CREATE TABLE IF NOT EXISTS categories (
    category        TEXT PRIMARY KEY,
    parent_category TEXT,
    source_scope    TEXT NOT NULL DEFAULT 'root',    -- 'root' | 'recursive'
    fetched_at      TEXT,
    fetch_status    TEXT NOT NULL DEFAULT 'pending', -- 'pending' | 'ok' | 'error'
    error           TEXT
);

-- Raw image manifest scraped from Wikimedia Commons.
-- One row per (series, category, file); images can appear in multiple categories.
CREATE TABLE IF NOT EXISTS images (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Label provenance
    series              TEXT NOT NULL,
    wiki_title          TEXT,
    power_type          TEXT,               -- EMU | DMU | Electric Locomotive | Diesel Locomotive | Steam Locomotive | Electro-diesel Multiple Unit
    operator_en_json    TEXT NOT NULL,      -- JSON array, e.g. ["JR East"]

    -- Commons location
    root_category       TEXT NOT NULL,
    category            TEXT NOT NULL,
    category_path_json  TEXT NOT NULL DEFAULT '[]', -- JSON array: path from root_category to this category
    file_title          TEXT NOT NULL,
    pageid              INTEGER,

    -- Image metadata from Commons imageinfo API
    image_url           TEXT,
    thumb_url           TEXT,
    mime                TEXT,
    width               INTEGER,
    height              INTEGER,
    size                INTEGER,
    sha1                TEXT,
    extmetadata_json    TEXT,

    -- Filtering state (keyword filter, SigLIP2 classifier)
    excluded            INTEGER NOT NULL DEFAULT 0,
    exclude_reason      TEXT,               -- e.g. "interior" | "file:seat" | "category:parts"
    siglip_processed    INTEGER NOT NULL DEFAULT 0,

    -- Download state
    download_status     TEXT NOT NULL DEFAULT 'not_started', -- 'not_started' | 'downloaded' | 'failed' | 'missing_url'
    downloaded_path     TEXT,               -- relative to path.data_root
    fetched_at          TEXT NOT NULL,

    -- LLM-extracted metadata from category_path (img_filter_v2 step 3)
    submodel            TEXT,               -- e.g. "E231-500"
    bandai              TEXT,               -- 番台, e.g. "500"
    operator_en         TEXT,
    operator_jp         TEXT,
    special_formation   TEXT,
    special_livery      TEXT,

    -- Fine-grained label after manual subtype splitting (stage_08 fine-grained step)
    fine_grained_series TEXT,              -- e.g. "E233系-2000番台"; NULL → copy series at training time

    UNIQUE (series, category, file_title)
);

-- Many-to-many: file ↔ category memberships across the crawl depth.
CREATE TABLE IF NOT EXISTS image_categories (
    file_title   TEXT NOT NULL,
    category     TEXT NOT NULL,
    source_scope TEXT NOT NULL DEFAULT 'root', -- 'root' | 'recursive'
    PRIMARY KEY (file_title, category)
);

-- Bounding-box crops detected by Grounding-DINO.
-- Only bbox metadata is stored; actual crops are generated on-the-fly at training time.
CREATE TABLE IF NOT EXISTS crops (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id            INTEGER NOT NULL REFERENCES images(id),

    -- Denormalized snapshot for fast queries without JOIN
    series              TEXT,
    power_type          TEXT,

    -- Detection result
    crop_index          INTEGER NOT NULL,
    source_result_index INTEGER,
    detector_model      TEXT NOT NULL,      -- e.g. "IDEA-Research/grounding-dino-base"
    detector_label      TEXT,
    detector_score      REAL,

    -- Bounding box in original image pixel coords
    box_x1              REAL NOT NULL,
    box_y1              REAL NOT NULL,
    box_x2              REAL NOT NULL,
    box_y2              REAL NOT NULL,
    box_area            REAL NOT NULL,
    nms_iou_threshold   REAL NOT NULL,

    -- Review / noise-filter state (Small Loss Trick output)
    crop_status         TEXT NOT NULL DEFAULT 'pending', -- 'pending' | 'ok' | 'rejected'
    crop_reason         TEXT,

    created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    noise_score_v1     REAL,-- Small Loss Trick noise score v1 (higher is more likely to be noise)

    -- Manual review state for noise-analysis sampling UI
    noise_review_label     TEXT, -- 'ok' | 'wrong_label' | 'out_of_label_space' | 'bad_crop' | 'ambiguous'
    noise_review_note      TEXT,
    noise_reviewed_at      TEXT,
    noise_review_score_col TEXT,

    UNIQUE (image_id, detector_model, nms_iou_threshold, crop_index)
);

CREATE INDEX IF NOT EXISTS idx_crops_image_id   ON crops(image_id);
CREATE INDEX IF NOT EXISTS idx_crops_series     ON crops(series);
CREATE INDEX IF NOT EXISTS idx_crops_power_type ON crops(power_type);
CREATE INDEX IF NOT EXISTS idx_crops_detector   ON crops(detector_model, nms_iou_threshold);
