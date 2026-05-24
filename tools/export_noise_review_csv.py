from __future__ import annotations

"""Export human noise-review labels as a portable CSV overlay.

The output path passed by ``--output-csv-path`` is resolved under
``path.data_root`` unless it is absolute. The exported rows include stable
image/crop keys and bbox coordinates so they can later be imported on another
machine without relying on autoincrement ids.
"""

import argparse
import sys
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


def export_review_csv(*, config: dict, output_csv_path: Path) -> int:
    import pandas as pd
    from pipeline import utils

    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    query = """
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
            c.detector_score,
            c.noise_review_label,
            c.noise_review_note,
            c.noise_reviewed_at,
            c.noise_review_score_col
        FROM crops c
        JOIN images i ON i.id = c.image_id
        WHERE c.noise_review_label IS NOT NULL
          AND c.noise_review_label != ''
        ORDER BY i.series, i.file_title, c.detector_model, c.nms_iou_threshold, c.crop_index
    """
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        review_df = pd.read_sql_query(query, conn)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    review_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    return len(review_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export portable noise review labels from the local Wakareeru DB."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline_config.yaml. Defaults to config/pipeline_config.yaml.",
    )
    parser.add_argument(
        "--output-csv-path",
        required=True,
        help="Output CSV path. Relative paths are resolved under path.data_root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from pipeline import utils

    config = utils.load_pipeline_config(args.config)
    utils.init_db(config=config)
    output_csv_path = utils.join_data_root(args.output_csv_path, config=config)
    count = export_review_csv(config=config, output_csv_path=output_csv_path)
    print(f"Exported {count} reviewed crops to {output_csv_path}")


if __name__ == "__main__":
    main()
