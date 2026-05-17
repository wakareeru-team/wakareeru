import csv
import os
import sqlite3
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm

from PIL import Image

import constants
import utils

#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #暂时解决windows anaconda自带一个库导致冲突

config = utils.load_pipeline_config()
logger = utils.get_logger("stage_08_siglip_crop_filtering")
IMAGE_DB_PATH = utils.join_data_root(config["path"]["db_path"], config=config)
SIGLIP_CROP_VIEW_CANDIDATES = constants.SIGLIP_CROP_FILTER_CANDIDATES
SIGLIP_PROMPT_TO_LABEL = constants.SIGLIP_CROP_PROMPT_TO_LABEL
KEEP_LABELS = {"train"}


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
    top["label"] = SIGLIP_PROMPT_TO_LABEL.get(top["label"], top["label"])
    return top


def _pending_siglip_where(include_excluded: bool) -> str:
    filters = [
        "i.download_status = ?",
        "i.downloaded_path IS NOT NULL",
        "i.downloaded_path != ''",
        "c.siglip_filtered = 0",
        "c.crop_status = ?",
    ]
    if not include_excluded:
        filters.append("i.excluded = 0")
    return " AND ".join(filters)


def _count_pending_crops(
    conn: sqlite3.Connection,
    include_excluded: bool,
    total_limit: int,
) -> int:
    where_sql = _pending_siglip_where(include_excluded)
    total = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM crops c
        JOIN images i ON i.id = c.image_id
        WHERE {where_sql}
        """,
        (constants.DOWNLOAD_STATUS_DOWNLOADED, constants.CROP_STATUS_PENDING),
    ).fetchone()[0]
    return min(total, total_limit) if total_limit > 0 else total


def _fetch_pending_crop_batch(
    conn: sqlite3.Connection,
    last_id: int,
    batch_size: int,
    include_excluded: bool,
) -> list[dict]:
    where_sql = _pending_siglip_where(include_excluded)
    rows = conn.execute(
        f"""
        SELECT
            c.id AS crop_id,
            c.image_id,
            c.crop_index,
            c.detector_score,
            c.box_x1, c.box_y1, c.box_x2, c.box_y2,
            i.file_title,
            i.downloaded_path
        FROM crops c
        JOIN images i ON i.id = c.image_id
        WHERE c.id > ?
          AND {where_sql}
        ORDER BY c.id
        LIMIT ?
        """,
        (last_id, constants.DOWNLOAD_STATUS_DOWNLOADED, constants.CROP_STATUS_PENDING, batch_size),
    ).fetchall()
    return [dict(row) for row in rows]


def _image_abs_path(downloaded_path: str, config: dict | None = None) -> Path:
    return utils.join_data_root(str(downloaded_path).replace("\\", "/"), config=config)


def _resize_crop_for_pipeline(
    crop: Image.Image,
    resize: int | tuple[int, int] | None,
) -> Image.Image:
    if resize is None:
        return crop
    size = (resize, resize) if isinstance(resize, int) else tuple(resize)
    return crop.resize(size, Image.Resampling.BICUBIC)


def _load_existing_crops(
    rows: list[dict],
    config: dict | None,
    pad_frac: float,
    resize: int | tuple[int, int] | None,
) -> tuple[list[dict], list, list[dict]]:
    existing_rows = []
    missing_rows = []
    crops = []

    for row in rows:
        image_path = _image_abs_path(row["downloaded_path"], config=config)
        if not image_path.exists():
            logger.warning(
                "Missing source image, skipping: crop_id=%d image_id=%d %s",
                row["crop_id"],
                row["image_id"],
                row["downloaded_path"],
            )
            missing_rows.append(row)
            continue

        crop = utils.load_crop(row, config=config, pad_frac=pad_frac)
        crop = _resize_crop_for_pipeline(crop, resize=resize)
        existing_rows.append(row)
        crops.append(crop)

    return existing_rows, crops, missing_rows


def _chunks(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def _updates_from_outputs(rows: list[dict], outputs: list, threshold: float) -> tuple[list[tuple], list[dict]]:
    update_rows = []
    log_rows = []

    for row, result in zip(rows, outputs):
        top_result = siglip_top_filtered_to_label(result, threshold=threshold)
        label = top_result["label"]
        crop_status = constants.CROP_STATUS_OK if label in KEEP_LABELS else constants.CROP_STATUS_REJECTED
        crop_reason = None if crop_status == constants.CROP_STATUS_OK else f"siglip:{label}"

        update_rows.append((crop_status, crop_reason, 1, row["crop_id"]))
        log_rows.append(
            {
                "crop_id": row["crop_id"],
                "image_id": row["image_id"],
                "crop_index": row["crop_index"],
                "file_title": row["file_title"],
                "downloaded_path": row["downloaded_path"],
                "label": label,
                "score": top_result["score"],
                "crop_status": crop_status,
                "crop_reason": crop_reason,
                "detector_score": row["detector_score"],
            }
        )

    return update_rows, log_rows


def _updates_from_missing(rows: list[dict]) -> tuple[list[tuple], list[dict]]:
    update_rows = []
    log_rows = []

    for row in rows:
        update_rows.append((constants.CROP_STATUS_REJECTED, "missing_file", 1, row["crop_id"]))
        log_rows.append(
            {
                "crop_id": row["crop_id"],
                "image_id": row["image_id"],
                "crop_index": row["crop_index"],
                "file_title": row["file_title"],
                "downloaded_path": row["downloaded_path"],
                "label": "missing_file",
                "score": None,
                "crop_status": constants.CROP_STATUS_REJECTED,
                "crop_reason": "missing_file",
                "detector_score": row["detector_score"],
            }
        )

    return update_rows, log_rows


def _append_siglip_log(log_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "crop_id",
        "image_id",
        "crop_index",
        "file_title",
        "downloaded_path",
        "label",
        "score",
        "crop_status",
        "crop_reason",
        "detector_score",
    ]
    write_header = not os.path.exists(log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)



def batch_siglip_crop_classification(
    image_classifier,
    db_path: str = IMAGE_DB_PATH,
    config: dict | None = None,
    db_load_batch_size: int = 100,
    siglip_batch_size: int = 16,
    total_limit: int = -1,
    include_excluded: bool = False,
    threshold: float = 0.0,
    pad_frac: float = 0.04,
    resize: int | tuple[int, int] | None = None,
    log_path: str | None = None,
) -> dict[str, int]:
    """Run SigLIP zero-shot filtering for pending Grounding-DINO crops."""
    counts: Counter[str] = Counter()
    processed = 0
    last_id = 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        progress_total = _count_pending_crops(conn, include_excluded, total_limit)
        if progress_total == 0:
            logger.warning("No crops need SigLIP classification.")
            return {}

        with tqdm(total=progress_total, desc="SigLIP crop filtering", unit="crop") as pbar:
            while processed < progress_total:
                current_batch_size = min(db_load_batch_size, progress_total - processed)
                rows = _fetch_pending_crop_batch(
                    conn,
                    last_id=last_id,
                    batch_size=current_batch_size,
                    include_excluded=include_excluded,
                )
                if not rows:
                    break

                last_id = rows[-1]["crop_id"]
                existing_rows, crops, missing_rows = _load_existing_crops(
                    rows,
                    config=config,
                    pad_frac=pad_frac,
                    resize=resize,
                )

                if missing_rows:
                    update_rows, log_rows = _updates_from_missing(missing_rows)
                    conn.executemany(
                        """
                        UPDATE crops
                        SET crop_status = ?,
                            crop_reason = ?,
                            siglip_filtered = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        update_rows,
                    )
                    if log_path:
                        _append_siglip_log(log_path, log_rows)
                    for row in log_rows:
                        counts[row["label"]] += 1
                    conn.commit()
                    processed += len(missing_rows)
                    pbar.update(len(missing_rows))
                    pbar.set_postfix(dict(counts))

                for start, crop_batch in _chunks(crops, max(1, siglip_batch_size)):
                    row_batch = existing_rows[start:start + len(crop_batch)]
                    outputs = image_classifier(crop_batch, SIGLIP_CROP_VIEW_CANDIDATES)
                    update_rows, log_rows = _updates_from_outputs(row_batch, outputs, threshold)
                    conn.executemany(
                        """
                        UPDATE crops
                        SET crop_status = ?,
                            crop_reason = ?,
                            siglip_filtered = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        update_rows,
                    )
                    if log_path:
                        _append_siglip_log(log_path, log_rows)
                    for row in log_rows:
                        counts[row["label"]] += 1

                    conn.commit()
                    processed += len(row_batch)
                    pbar.update(len(row_batch))
                    pbar.set_postfix(dict(counts))

                logger.info("Processed %d/%d crops: %s", processed, progress_total, dict(counts))

    return dict(counts)




def main(config_override: dict | None = None):
    '''由于crop出来仍然看到一些不属于火车的图片，所以在这里对crop level再做一次筛选'''
    
    import torch
    from dotenv import load_dotenv
    from huggingface_hub import login
    from transformers import pipeline
    
    device = "mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu"


    load_dotenv(override=True)
    cfg = config_override or config
    utils.init_db(config=cfg)

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
    
    crop_config = cfg['crop_filtering']
    image_classifier = pipeline(
        model=crop_config["siglip_model_name"],
        task="zero-shot-image-classification",
        use_fast=True,
        batch_size=int(crop_config.get("siglip_batch_size", 16)),
        device=device,
    )
    logger.info("SigLIP pipeline 加速设备: %s", image_classifier.device)

    log_path = crop_config.get("siglip_log_path")
    if log_path:
        log_path = utils.join_data_root(log_path, config=cfg)
    db_path = utils.join_data_root(cfg["path"]["db_path"], config=cfg)

    result = batch_siglip_crop_classification(
        image_classifier=image_classifier,
        db_path=db_path,
        config=cfg,
        db_load_batch_size=int(crop_config.get("siglip_db_load_batch_size", 100)),
        siglip_batch_size=int(crop_config.get("siglip_batch_size", 16)),
        total_limit=int(crop_config.get("siglip_total_limit", -1)),
        include_excluded=_as_bool(crop_config.get("siglip_include_excluded"), default=False),
        threshold=float(crop_config.get("siglip_threshold", 0.0)),
        pad_frac=float(crop_config.get("crop_pad_frac", 0.04)),
        resize=crop_config.get("crop_resize"),
        log_path=log_path,
    )
    logger.info("SigLIP crop result: %s", result)


if __name__ == "__main__":
    main()
    
