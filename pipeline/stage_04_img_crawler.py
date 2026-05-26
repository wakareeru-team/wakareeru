import csv
import os
import random
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping
from urllib.parse import unquote

import httpx
from tqdm import tqdm

import constants
import path_normalization
import utils


config = utils.load_pipeline_config()

logger = utils.get_logger("stage_04_img_crawler")
IMAGE_DB_PATH = utils.join_data_root(config["path"]["db_path"], config=config)
IMG_ROOT = utils.join_data_root(config["path"]["raw_img_dir"], config=config)


def safe_path_component(value: str, max_len: int = 120) -> str:
    """Sanitize one path component while keeping readable train/category names."""
    value = unquote(str(value or "")).removeprefix("File:").strip()
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return path_normalization.normalize_text(value or "unnamed")[:max_len]


def local_image_path(
    record: Mapping[str, Any],
    img_root: str | os.PathLike = IMG_ROOT,
    data_root: str | os.PathLike | None = None,
) -> tuple[str, str]:
    """Return absolute path and data-root-relative path for one downloaded image."""
    series_dir = safe_path_component(record.get("series"))
    file_name = safe_path_component(record.get("file_title"))
    sha1 = (record.get("sha1") or "")[:6]
    if sha1:
        file_name = f"{sha1}_{file_name}"

    abs_path = os.path.join(img_root, series_dir, file_name)
    data_root = data_root or utils.get_data_root()
    rel_path = path_normalization.normalize_rel_path(
        os.path.relpath(abs_path, data_root).replace(os.sep, "/")
    )
    return abs_path, rel_path


def _download_retry_sleep(attempt: int, response: httpx.Response | None = None) -> float:
    """Return a polite retry delay, honoring Retry-After when Commons sends it."""
    if response is not None and response.headers.get("Retry-After"):
        try:
            return min(float(response.headers["Retry-After"]), 30.0)
        except ValueError:
            pass
    return min(1.5 * (2**attempt) + random.uniform(0, 0.5), 30.0)


def download_one_image(
    record: Mapping[str, Any],
    img_root: str | os.PathLike = IMG_ROOT,
    data_root: str | os.PathLike | None = None,
    max_retries: int = 3,
    request_interval: float = 0.1,
) -> dict:
    """Download one manifest image and return the SQLite update payload."""
    time.sleep(request_interval)
    image_url = record.get("image_url")
    error = None
    if not image_url:
        status, rel_path, error = constants.DOWNLOAD_STATUS_MISSING_URL, None, "image_url is empty"
    else:
        abs_path, rel_path = local_image_path(record, img_root=img_root, data_root=data_root)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        if os.path.exists(abs_path) and os.path.getsize(abs_path) > 0:
            status = constants.DOWNLOAD_STATUS_DOWNLOADED
        else:
            status = constants.DOWNLOAD_STATUS_FAILED
            temp_path = f"{abs_path}.part.{record['id']}"
            for attempt in range(max_retries + 1):
                try:
                    with httpx.stream(
                        "GET",
                        image_url,
                        headers=constants.COMMONS_HEADERS,
                        timeout=60,
                        follow_redirects=True,
                    ) as resp:
                        resp.raise_for_status()
                        with open(temp_path, "wb") as f:
                            for chunk in resp.iter_bytes():
                                if chunk:
                                    f.write(chunk)
                    os.replace(temp_path, abs_path)
                    status, error = constants.DOWNLOAD_STATUS_DOWNLOADED, None
                    break

                except httpx.HTTPStatusError as exc:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    code = exc.response.status_code
                    error = f"HTTP {code}: {exc.response.reason_phrase}"
                    retryable = code in {429, 502, 503, 504}
                    if retryable and attempt < max_retries:
                        sleep_s = _download_retry_sleep(attempt, exc.response)
                        logger.info(
                            "Retrying %s: %s, retry in %.1fs (%d/%d)",
                            record["file_title"],
                            error,
                            sleep_s,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(sleep_s)
                        continue
                    logger.error("Failed %s: %s", record["file_title"], error)
                    rel_path = None
                    break

                except httpx.HTTPError as exc:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < max_retries:
                        sleep_s = _download_retry_sleep(attempt)
                        logger.info(
                            "Retrying %s: %s, retry in %.1fs (%d/%d)",
                            record["file_title"],
                            error,
                            sleep_s,
                            attempt + 1,
                            max_retries,
                        )
                        time.sleep(sleep_s)
                        continue
                    logger.error("Failed %s: %s", record["file_title"], error)
                    rel_path = None
                    break

    return {
        "id": record["id"],
        "file_title": record["file_title"],
        "status": status,
        "path": rel_path,
        "error": error,
    }


def _update_download_results(conn: sqlite3.Connection, results: list[dict]) -> None:
    if not results:
        return
    conn.executemany(
        """
        UPDATE images
        SET download_status = ?, downloaded_path = ?
        WHERE id = ?
        """,
        [(result["status"], result["path"], result["id"]) for result in results],
    )


def _pending_download_where(retry_failed: bool) -> tuple[str, tuple[str]]:
    if retry_failed:
        return "download_status != ?", (constants.DOWNLOAD_STATUS_DOWNLOADED,)
    return "download_status = ?", (constants.DOWNLOAD_STATUS_NOT_STARTED,)


def _count_pending_downloads(
    conn: sqlite3.Connection,
    retry_failed: bool,
    total_limit: int,
) -> int:
    status_filter, params = _pending_download_where(retry_failed)
    total = conn.execute(
        f"""
        SELECT COUNT(*)
        FROM images
        WHERE excluded = 0
          AND image_url IS NOT NULL
          AND {status_filter}
        """,
        params,
    ).fetchone()[0]
    return min(total, total_limit) if total_limit > 0 else total


def _fetch_pending_download_batch(
    conn: sqlite3.Connection,
    last_id: int,
    batch_size: int,
    retry_failed: bool,
) -> list[dict]:
    status_filter, params = _pending_download_where(retry_failed)
    rows = conn.execute(
        f"""
        SELECT id, series, category, file_title, image_url, sha1
        FROM images
        WHERE id > ?
          AND excluded = 0
          AND image_url IS NOT NULL
          AND {status_filter}
        ORDER BY id
        LIMIT ?
        """,
        (last_id, *params, batch_size),
    ).fetchall()
    return [dict(row) for row in rows]


def _append_download_log(log_path: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = ["id", "file_title", "status", "path", "error"]
    write_header = not os.path.exists(log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    if not write_header:
        with open(log_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header:
            fieldnames = existing_header
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def batch_download_manifest_from_db(
    db_path: str = IMAGE_DB_PATH,
    img_root: str | os.PathLike = IMG_ROOT,
    data_root: str | os.PathLike | None = None,
    batch_size: int = 500,
    total_limit: int = -1,
    retry_failed: bool = True,
    log_path: str | None = None,
    max_retries: int = 3,
    workers: int = 1,
    request_interval: float = 0.1,
) -> dict[str, int]:
    """Download pending images in fixed-size DB batches with tqdm progress."""
    counts: Counter[str] = Counter()
    processed = 0
    last_id = 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        progress_total = _count_pending_downloads(conn, retry_failed, total_limit)
        if progress_total == 0:
            logger.warning("没有图片需要下载。")
            return {}

        with tqdm(total=progress_total, desc="Downloading images", unit="img") as pbar:
            while processed < progress_total:
                current_batch_size = min(batch_size, progress_total - processed)
                rows = _fetch_pending_download_batch(
                    conn,
                    last_id=last_id,
                    batch_size=current_batch_size,
                    retry_failed=retry_failed,
                )
                if not rows:
                    break

                last_id = max(last_id, max(int(record["id"]) for record in rows))
                results = []
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(
                            download_one_image,
                            record,
                            img_root=img_root,
                            data_root=data_root,
                            max_retries=max_retries,
                            request_interval=request_interval,
                        )
                        for record in rows
                    ]
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        counts[result["status"]] += 1
                        processed += 1
                        pbar.update(1)
                        pbar.set_postfix(dict(counts))

                _update_download_results(conn, results)
                conn.commit()
                if log_path:
                    _append_download_log(log_path, results)
                logger.info("Processed %d/%d images: %s", processed, progress_total, dict(counts))

    return dict(counts)


def main(config: dict | None = None):
    config = config if config is not None else utils.load_pipeline_config()
    utils.init_db(config=config)
    img_root = utils.join_data_root(config["path"]["raw_img_dir"], config=config)
    data_root = utils.get_data_root(config)
    os.makedirs(img_root, exist_ok=True)

    crawler_config = config["crawler"]
    log_path = config["path"]["img_crawl_log_path"]
    if log_path:
        log_path = utils.join_data_root(log_path, config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)

    result = batch_download_manifest_from_db(
        db_path=db_path,
        img_root=img_root,
        data_root=data_root,
        batch_size=int(crawler_config["download_batch_size"]),
        total_limit=int(crawler_config["download_total_limit"]),
        retry_failed=_as_bool(crawler_config["download_retry_failed"]),
        log_path=log_path,
        max_retries=int(crawler_config["download_max_retries"]),
        workers=int(crawler_config["download_workers"]),
        request_interval=float(crawler_config["download_request_interval"]),
    )
    logger.info("下载结果: %s", result)

    normalization_report = path_normalization.normalize_downloaded_image_paths(
        db_path=db_path,
        img_root=img_root,
    )
    logger.info(
        "图片路径 NFC 规范化完成: filesystem_renames=%d, db_updates=%d",
        normalization_report.filesystem_renames,
        normalization_report.db_updates,
    )

if __name__ == "__main__":
    main()
