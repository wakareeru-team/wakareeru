import csv
import os
import sqlite3
from collections import Counter
from tqdm.auto import tqdm

import constants
import utils

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #暂时解决windows anaconda自带一个库导致冲突

config = utils.load_pipeline_config()
PROJECT_ROOT = utils.get_project_root()
logger = utils.get_logger("stage_05_siglip_image_filtering")
IMAGE_DB_PATH = utils.join_root_path(config["path"]["db_path"])

SIGLIP_VIEW_CANDIDATES = constants.SIGLIP_VIEW_CANDIDATES
SIGLIP_PROMPT_TO_LABEL = constants.SIGLIP_PROMPT_TO_LABEL
KEEP_LABELS = {"exterior", "uncertain"}


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def siglip_top_filtered_to_label(result: list[dict] | dict, threshold: float = 0.0) -> dict:
    """Return the normalized top zero-shot label from a SigLIP pipeline result."""
    top = result[0].copy() if isinstance(result, list) else result.copy()
    if top["score"] < threshold:
        top["label"] = "uncertain"
    top["label"] = SIGLIP_PROMPT_TO_LABEL[top["label"]]
    return top


def _pending_siglip_where(include_excluded: bool) -> str:
    filters = [
        "download_status = ?",
        "downloaded_path IS NOT NULL",
        "downloaded_path != ''",
        "siglip_processed = 0",
    ]
    if not include_excluded:
        filters.append("excluded = 0")
    return " AND ".join(filters)


def _count_pending_images(
    conn: sqlite3.Connection,
    include_excluded: bool,
    total_limit: int,
) -> int:
    where_sql = _pending_siglip_where(include_excluded)
    total = conn.execute(
        f"SELECT COUNT(*) FROM images WHERE {where_sql}",
        (constants.DOWNLOAD_STATUS_DOWNLOADED,),
    ).fetchone()[0]
    return min(total, total_limit) if total_limit > 0 else total


def _fetch_pending_image_batch(
    conn: sqlite3.Connection,
    last_id: int,
    batch_size: int,
    include_excluded: bool,
) -> list[dict]:
    where_sql = _pending_siglip_where(include_excluded)
    rows = conn.execute(
        f"""
        SELECT id, file_title, downloaded_path
        FROM images
        WHERE id > ?
          AND {where_sql}
        ORDER BY id
        LIMIT ?
        """,
        (last_id, constants.DOWNLOAD_STATUS_DOWNLOADED, batch_size),
    ).fetchall()
    return [dict(row) for row in rows]


def _image_abs_path(downloaded_path: str) -> str:
    return utils.join_root_path(os.path.join("data", downloaded_path))


def _updates_from_outputs(rows: list[dict], outputs: list, threshold: float) -> tuple[list[tuple], list[dict]]:
    update_rows = []
    log_rows = []

    for row, result in zip(rows, outputs):
        top_result = siglip_top_filtered_to_label(result, threshold=threshold)
        label = top_result["label"]
        excluded = 0 if label in KEEP_LABELS else 1
        exclude_reason = None if excluded == 0 else label

        update_rows.append((excluded, exclude_reason, 1, row["id"]))
        log_rows.append(
            {
                "id": row["id"],
                "file_title": row["file_title"],
                "label": label,
                "score": top_result["score"],
                "excluded": excluded,
                "exclude_reason": exclude_reason,
            }
        )

    return update_rows, log_rows


def _append_siglip_log(log_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = ["id", "file_title", "label", "score", "excluded", "exclude_reason"]
    write_header = not os.path.exists(log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)



def batch_siglip_classification(
    image_classifier,
    db_path: str = IMAGE_DB_PATH,
    db_load_batch_size: int = 100,
    total_limit: int = -1,
    include_excluded: bool = False,
    threshold: float = 0.0,
    log_path: str | None = None,
) -> dict[str, int]:
    """Run SigLIP zero-shot filtering for downloaded, unprocessed images."""
    counts: Counter[str] = Counter()
    processed = 0
    last_id = 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        progress_total = _count_pending_images(conn, include_excluded, total_limit)
        if progress_total == 0:
            logger.warning("No images need SigLIP classification.")
            return {}

        with tqdm(total=progress_total, desc="SigLIP filtering", unit="img") as pbar:
            while processed < progress_total:
                current_batch_size = min(db_load_batch_size, progress_total - processed)
                rows = _fetch_pending_image_batch(
                    conn,
                    last_id=last_id,
                    batch_size=current_batch_size,
                    include_excluded=include_excluded,
                )
                if not rows:
                    break

                last_id = rows[-1]["id"]
                existing_rows = [row for row in rows if os.path.exists(_image_abs_path(row["downloaded_path"]))]
                existing_ids = {row["id"] for row in existing_rows}
                missing_rows = [row for row in rows if row["id"] not in existing_ids]

                if missing_rows:
                    for row in missing_rows:
                        logger.warning("Missing file, skipping: id=%d %s", row["id"], row["downloaded_path"])
                    counts["missing_file"] += len(missing_rows)

                if existing_rows:
                    paths = [_image_abs_path(row["downloaded_path"]) for row in existing_rows]
                    outputs = image_classifier(paths, SIGLIP_VIEW_CANDIDATES)
                    update_rows, log_rows = _updates_from_outputs(existing_rows, outputs, threshold)
                    conn.executemany(
                        """
                        UPDATE images
                        SET excluded = ?,
                            exclude_reason = ?,
                            siglip_processed = ?
                        WHERE id = ?
                        """,
                        update_rows,
                    )
                    if log_path:
                        _append_siglip_log(log_path, log_rows)
                    for row in log_rows:
                        counts[row["label"]] += 1

                conn.commit()
                processed += len(rows)
                pbar.update(len(rows))
                pbar.set_postfix(dict(counts))
                logger.info("Processed %d/%d images: %s", processed, progress_total, dict(counts))

    return dict(counts)


def main(config_override: dict | None = None):
    import torch
    from dotenv import load_dotenv
    from huggingface_hub import login
    from transformers import pipeline

    load_dotenv(override=True)
    cfg = config_override or config
    utils.init_db()

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        login(token=token)
        logger.info("Logged into Hugging Face Hub.")
    else:
        logger.warning("HUGGINGFACEHUB_API_TOKEN not found in environment variables." 
                       "Further actions may fail if the model requires authentication.")

    logger.info(
        "MPS available: %s; CUDA available: %s; CUDA devices: %d",
        torch.backends.mps.is_available(),
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )

    image_filtering_config = cfg["image_filtering"]
    image_classifier = pipeline(
        model=image_filtering_config["siglip_model_name"],
        task="zero-shot-image-classification",
        use_fast=True,
        batch_size=int(image_filtering_config.get("siglip_batch_size", 16)),
    )

    log_path = image_filtering_config.get("siglip_log_path")
    if log_path:
        log_path = utils.join_root_path(log_path)

    result = batch_siglip_classification(
        image_classifier=image_classifier,
        db_path=utils.join_root_path(cfg["path"]["db_path"]),
        db_load_batch_size=int(image_filtering_config.get("siglip_db_load_batch_size", 100)),
        total_limit=int(image_filtering_config.get("siglip_total_limit", -1)),
        include_excluded=_as_bool(image_filtering_config.get("siglip_include_excluded"), default=False),
        threshold=float(image_filtering_config.get("siglip_threshold", 0.0)),
        log_path=log_path,
    )
    logger.info("SigLIP result: %s", result)


if __name__ == "__main__":
    main()
