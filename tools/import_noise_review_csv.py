from __future__ import annotations

"""Import human noise-review labels from a portable CSV overlay.

The CSV path passed by ``--review-csv-path`` is resolved under ``path.data_root``
unless it is absolute. Rows are matched without relying on autoincrement ids,
using stable image/crop keys plus a bbox IoU sanity check.

Required CSV columns:
series, file_title, root_category, category_path_json, detector_model,
nms_iou_threshold, crop_index, box_x1, box_y1, box_x2, box_y2,
noise_review_label.
"""

import argparse
import logging
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Project root not found from {start}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import constants  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("import_noise_review_csv")

REVIEW_COLUMN_DEFS = {
    "noise_review_label": "TEXT",
    "noise_review_note": "TEXT",
    "noise_reviewed_at": "TEXT",
    "noise_review_score_col": "TEXT",
}

STABLE_KEY_COLUMNS = [
    "series",
    "file_title",
    "root_category",
    "category_path_json",
    "detector_model",
    "nms_iou_threshold",
    "crop_index",
]

BBOX_COLUMNS = ["box_x1", "box_y1", "box_x2", "box_y2"]
REQUIRED_COLUMNS = [*STABLE_KEY_COLUMNS, *BBOX_COLUMNS, "noise_review_label"]


def quote_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQLite identifier: {name!r}")
    return f'"{name}"'


def ensure_review_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(crops)").fetchall()}
    for col, col_type in REVIEW_COLUMN_DEFS.items():
        if col not in cols:
            conn.execute(f"ALTER TABLE crops ADD COLUMN {quote_ident(col)} {col_type}")


def normalize_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["series", "file_title", "root_category", "category_path_json", "detector_model"]:
        df[col] = df[col].fillna("").astype(str)
    df["nms_iou_threshold_key"] = df["nms_iou_threshold"].astype(float).map(lambda value: f"{value:.10g}")
    df["crop_index_key"] = df["crop_index"].astype(int).astype(str)
    return df


def key_columns_for_join() -> list[str]:
    return [
        "series",
        "file_title",
        "root_category",
        "category_path_json",
        "detector_model",
        "nms_iou_threshold_key",
        "crop_index_key",
    ]


def bbox_iou(row: pd.Series) -> float:
    import numpy as np

    src = np.array([row[f"{col}_csv"] for col in BBOX_COLUMNS], dtype=float)
    dst = np.array([row[f"{col}_db"] for col in BBOX_COLUMNS], dtype=float)
    ix1 = max(src[0], dst[0])
    iy1 = max(src[1], dst[1])
    ix2 = min(src[2], dst[2])
    iy2 = min(src[3], dst[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    src_area = max(0.0, src[2] - src[0]) * max(0.0, src[3] - src[1])
    dst_area = max(0.0, dst[2] - dst[0]) * max(0.0, dst[3] - dst[1])
    denom = src_area + dst_area - inter
    return 0.0 if denom <= 0 else inter / denom


def load_review_csv(path: Path) -> pd.DataFrame:
    import pandas as pd

    review_df = pd.read_csv(path, dtype={"noise_review_label": str})
    missing = [col for col in REQUIRED_COLUMNS if col not in review_df.columns]
    if missing:
        raise ValueError(f"review CSV missing required columns: {missing}")

    review_df = review_df.dropna(subset=["noise_review_label"]).copy()
    review_df["noise_review_label"] = review_df["noise_review_label"].astype(str).str.strip()
    review_df = review_df[review_df["noise_review_label"] != ""].copy()
    invalid_labels = sorted(set(review_df["noise_review_label"]) - set(constants.NOISE_REVIEW_LABELS))
    if invalid_labels:
        raise ValueError(f"review CSV contains unknown labels: {invalid_labels}")

    if "noise_review_note" not in review_df.columns:
        review_df["noise_review_note"] = ""
    if "noise_reviewed_at" not in review_df.columns:
        review_df["noise_reviewed_at"] = ""
    if "noise_review_score_col" not in review_df.columns:
        review_df["noise_review_score_col"] = ""

    return normalize_key_frame(review_df)


def load_db_review_targets(conn: sqlite3.Connection) -> pd.DataFrame:
    import pandas as pd

    target_df = pd.read_sql_query(
        """
        SELECT
            c.id AS crop_id,
            i.series,
            i.file_title,
            i.root_category,
            i.category_path_json,
            c.detector_model,
            c.nms_iou_threshold,
            c.crop_index,
            c.box_x1,
            c.box_y1,
            c.box_x2,
            c.box_y2,
            c.noise_review_label AS existing_review_label
        FROM crops c
        JOIN images i ON i.id = c.image_id
        """,
        conn,
    )
    return normalize_key_frame(target_df)


def import_review_csv(
    *,
    db_path: Path,
    review_csv_path: Path,
    bbox_iou_min: float,
    overwrite_existing: bool,
    dry_run: bool,
) -> dict[str, int]:
    import numpy as np
    import pandas as pd

    review_df = load_review_csv(review_csv_path)
    review_df["_csv_row"] = np.arange(len(review_df))

    with sqlite3.connect(db_path) as conn:
        ensure_review_columns(conn)
        target_df = load_db_review_targets(conn)

        join_cols = key_columns_for_join()
        duplicate_csv = int(review_df.duplicated(join_cols, keep=False).sum())
        if duplicate_csv:
            logger.warning("review CSV 中有 %d 行 stable key 重复；将按文件顺序保留最后一行。", duplicate_csv)
            review_df = review_df.sort_values("_csv_row").drop_duplicates(join_cols, keep="last")

        target_counts = target_df.groupby(join_cols).size().rename("_target_count").reset_index()
        merged = review_df.merge(target_counts, on=join_cols, how="left")
        missing_mask = merged["_target_count"].isna()
        ambiguous_mask = merged["_target_count"].fillna(0).astype(int) > 1

        candidates = merged[~missing_mask & ~ambiguous_mask].merge(
            target_df,
            on=join_cols,
            how="inner",
            suffixes=("_csv", "_db"),
        )
        if not candidates.empty:
            candidates["bbox_iou"] = candidates.apply(bbox_iou, axis=1)
        bbox_mismatch_mask = candidates["bbox_iou"] < bbox_iou_min if not candidates.empty else pd.Series(dtype=bool)

        matched = candidates[~bbox_mismatch_mask].copy() if not candidates.empty else candidates
        existing_mask = (
            matched["existing_review_label"].notna()
            & (matched["existing_review_label"].astype(str).str.strip() != "")
        )
        skipped_existing = int(existing_mask.sum()) if not overwrite_existing else 0
        if not overwrite_existing:
            matched = matched[~existing_mask].copy()

        now = datetime.now(timezone.utc).isoformat()
        update_rows = [
            (
                row["noise_review_label"],
                "" if pd.isna(row["noise_review_note"]) else str(row["noise_review_note"]),
                now if pd.isna(row["noise_reviewed_at"]) or str(row["noise_reviewed_at"]).strip() == "" else str(row["noise_reviewed_at"]),
                "" if pd.isna(row["noise_review_score_col"]) else str(row["noise_review_score_col"]),
                int(row["crop_id"]),
            )
            for _, row in matched.iterrows()
        ]

        if update_rows and not dry_run:
            conn.executemany(
                """
                UPDATE crops
                SET noise_review_label = ?,
                    noise_review_note = ?,
                    noise_reviewed_at = ?,
                    noise_review_score_col = ?
                WHERE id = ?
                """,
                update_rows,
            )
            conn.commit()

    return {
        "csv_rows": int(len(review_df)),
        "matched": int(len(candidates)),
        "missing": int(missing_mask.sum()),
        "ambiguous": int(ambiguous_mask.sum()),
        "bbox_mismatch": int(bbox_mismatch_mask.sum()) if not candidates.empty else 0,
        "skipped_existing": skipped_existing,
        "updated": len(update_rows),
        "dry_run": int(dry_run),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import noise review labels from a stable-key CSV into the local Wakareeru DB."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline_config.yaml. Defaults to config/pipeline_config.yaml.",
    )
    parser.add_argument(
        "--review-csv-path",
        required=True,
        help="Review CSV path. Relative paths are resolved under path.data_root.",
    )
    parser.add_argument(
        "--bbox-iou-min",
        type=float,
        default=0.98,
        help="Minimum IoU between CSV bbox and DB bbox for accepting a match.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite crops that already have noise_review_label.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report import counts without writing to the database.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from pipeline import utils

    config = utils.load_pipeline_config(args.config)
    utils.init_db(config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    review_csv_path = utils.join_data_root(args.review_csv_path, config=config)

    report = import_review_csv(
        db_path=db_path,
        review_csv_path=review_csv_path,
        bbox_iou_min=float(args.bbox_iou_min),
        overwrite_existing=bool(args.overwrite_existing),
        dry_run=bool(args.dry_run),
    )
    logger.info("review CSV 导入报告: %s", report)


if __name__ == "__main__":
    main()
