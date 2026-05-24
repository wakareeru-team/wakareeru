import ast
import datetime
import json
import os
import random
import re
import time
import httpx
import pandas as pd
import ast
import sqlite3
from datetime import datetime, timezone

from tqdm.auto import tqdm

import constants
import utils

config = utils.load_pipeline_config()

COMMONS_MODEL_CSV = utils.join_data_root(config['path']['series_commons_path'], config=config)
logger = utils.get_logger("stage_03_manifest_crawling")
IMAGE_DB_PATH = utils.join_data_root(config['path']['db_path'], config=config)

# ========= Helper和一些格式映射 =========
def utc_now() -> str:
    """Return an ISO timestamp for manifest writes."""
    return datetime.now(timezone.utc).isoformat()


def parse_literal(value, default):
    """Parse list/dict values exported by pandas CSV round-trips."""
    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value) or value == "":
        return default
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return default


def map_power_type(type_value: str | None) -> str | None:
    """Map the Japanese rolling-stock type into the English power type taxonomy."""
    if pd.isna(type_value):
        return None
    return constants.POWER_TYPE_MAP.get(str(type_value).strip())


def load_commons_models(path: str = COMMONS_MODEL_CSV) -> pd.DataFrame:
    """Load model rows with Commons root metadata from CSV and mapped power_type."""
    df = pd.read_csv(path)
    for col in ["operator_page_title", "operator_jp", "operator_en", "commons_candidates"]:
        if col in df:
            df[col] = df[col].apply(lambda v: parse_literal(v, []))
    if "commons_operator_roots" in df:
        df["commons_operator_roots"] = df["commons_operator_roots"].apply(lambda v: parse_literal(v, {}))
    if "needs_review" in df:
        df["needs_review"] = df["needs_review"].apply(lambda v: str(v).strip().lower() == "true")
    if "type" in df:
        df["power_type"] = df["type"].apply(map_power_type)
    return df


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def select_models_to_crawl(
    models: pd.DataFrame,
    crawler_config: dict,
) -> tuple[pd.DataFrame, str]:
    """Select full or partial manifest crawl scope from pipeline config."""
    models = models.copy()
    models = models[models["commons_root_category"].notna()]
    models = models[models["commons_root_category"].astype(str).str.strip() != ""]
    if "needs_review" in models:
        models = models[models["needs_review"] != True]

    full_on = _as_bool(crawler_config["full_series_crawling"])
    if full_on:
        return models.reset_index(drop=True), "全量"

    series_scope = crawler_config["series_test_scope"]
    if isinstance(series_scope, str):
        series_scope = [series_scope]
    if not series_scope:
        logger.warning("未开启全量爬取，且 series_test_scope 为空；本次不会爬取任何车型")
        return models.iloc[0:0].copy(), "部分"

    selected = models[models["series"].isin(series_scope)].copy()
    missing = [series for series in series_scope if series not in set(selected["series"])]
    if missing:
        logger.warning("以下测试车型没有可用 Commons root，已跳过：%s", missing)
    return selected.reset_index(drop=True), "部分"

def _commons_query(params: dict, max_retries: int = 3, base_sleep: float = 1.0) -> dict | None:
    for attempt in range(max_retries + 1):
        try:
            resp = httpx.get(constants.COMMONS_API_URL, params=params, headers=constants.COMMONS_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                error = data["error"]
                raise RuntimeError(f'{error.get("code", "api-error")} {error.get("info", data)}')
            return data
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            if attempt >= max_retries:
                logger.error(f'Commons 请求失败：{params}（{exc}）')
                return None
            sleep_s = base_sleep * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(f'Commons 临时错误，{sleep_s:.1f} 秒后重试（{attempt + 1}/{max_retries}）')
            time.sleep(sleep_s)



def fetch_subcategories(category: str) -> list[str] | None:
    # 这里只取直接子 category，不取文件；后续 flatten 阶段再决定递归深度。
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "subcat",
        "cmlimit": "max",
        "format": "json",
    }
    subcats = []
    while True:
        data = _commons_query(params)
        if data is None:
            return None
        for item in data.get("query", {}).get("categorymembers", []):
            subcats.append(item["title"].removeprefix("Category:"))
        if "continue" not in data:
            return subcats
        params.update(data["continue"])


# ========= 图片格式过滤以及生成数据库 Entry =========

def has_excluded_pattern(text: str, patterns: tuple[str, ...]) -> str | None:
    """Return the first exclusion pattern found in text."""
    lowered = (text or "").lower()
    for pattern in patterns:
        if pattern.lower() in lowered:
            return pattern
    return None


def category_exclude_reason(row: pd.Series, category: str) -> str | None:
    """Return a category-level exclusion reason for this series/category pair."""
    category_exclude = has_excluded_pattern(category, constants.CATEGORY_EXCLUDE_PATTERNS)
    if category_exclude:
        return f"category:{category_exclude}"

    series_patterns = constants.SERIES_CATEGORY_EXCLUDE_PATTERNS.get(row["series"], ())
    series_exclude = has_excluded_pattern(category, series_patterns)
    if series_exclude:
        return f"category:wrong-series:{series_exclude}"

    return None


def fetch_category_file_members(category: str, max_files: int = 10) -> list[dict]:
    """Fetch direct file members of a Commons category."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "file",
        "cmlimit": min(max_files, 50),
        "format": "json",
    }
    files = []
    while len(files) < max_files:
        data = _commons_query(params)
        if data is None:
            break
        files.extend(data.get("query", {}).get("categorymembers", []))
        if "continue" not in data or len(files) >= max_files:
            break
        params.update(data["continue"])
        params["cmlimit"] = min(max_files - len(files), 50)
    return files[:max_files]


def fetch_imageinfo(file_titles: list[str], thumb_width: int = 512) -> dict[str, dict]:
    """Fetch image URLs, dimensions, mime, sha1, and Commons metadata."""
    result = {}
    for start in range(0, len(file_titles), 50):
        batch = file_titles[start:start + 50]
        data = _commons_query({
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|mime|size|sha1|extmetadata",
            "iiurlwidth": thumb_width,
            "format": "json",
        })
        if data is None:
            continue
        for page in data.get("query", {}).get("pages", {}).values():
            title = page.get("title")
            info = (page.get("imageinfo") or [{}])[0]
            result[title] = {"pageid": page.get("pageid"), **info}
    return result


def build_image_records(
    row: pd.Series, category: str, max_files: int = 10, category_path: list[str] | None = None
) -> list[dict]:
    """Build manifest records for files directly under one category."""
    category_path = category_path or [category]
    category_exclude = category_exclude_reason(row, category)
    members = fetch_category_file_members(category, max_files=max_files)
    titles = [m["title"] for m in members]
    info_by_title = fetch_imageinfo(titles)

    records = []
    for member in members:
        file_title = member["title"]
        info = info_by_title.get(file_title, {})
        exclude_reason = None
        if category_exclude:
            exclude_reason = category_exclude
        elif has_excluded_pattern(file_title, constants.FILE_INTERIOR_PATTERNS):
            exclude_reason = "interior"
        elif has_excluded_pattern(file_title, constants.FILE_DETAIL_PATTERNS):
            exclude_reason = "detail"

        records.append({
            "series": row["series"],
            "wiki_title": row.get("wiki_title"),
            "power_type": None if pd.isna(row.get("power_type")) else row.get("power_type"),
            "operator_en_json": json.dumps(row.get("operator_en", []), ensure_ascii=False),
            "root_category": row["commons_root_category"],
            "category": category,
            "category_path_json": json.dumps(category_path, ensure_ascii=False),
            "file_title": file_title,
            "pageid": info.get("pageid") or member.get("pageid"),
            "image_url": info.get("url"),
            "thumb_url": info.get("thumburl"),
            "mime": info.get("mime"),
            "width": info.get("width"),
            "height": info.get("height"),
            "size": info.get("size"),
            "sha1": info.get("sha1"),
            "extmetadata_json": json.dumps(info.get("extmetadata", {}), ensure_ascii=False),
            "excluded": int(exclude_reason is not None),
            "exclude_reason": exclude_reason,
            "fetched_at": utc_now(),
        })
    return records


# =============== MIME类型过滤 ===============

def purge_non_image_manifest_records(conn: sqlite3.Connection) -> int:
    """Delete existing DB rows whose MIME is missing or not supported raster image data."""
    conn.execute(
        """
        DELETE FROM image_categories
        WHERE EXISTS (
            SELECT 1
            FROM images
            WHERE images.file_title = image_categories.file_title
              AND images.category = image_categories.category
              AND (
                  LOWER(COALESCE(images.mime, '')) NOT LIKE 'image/%'
                  OR LOWER(COALESCE(images.mime, '')) = 'image/svg+xml'
              )
        )
        """
    )
    deleted = conn.execute(
        """
        DELETE FROM images
        WHERE LOWER(COALESCE(mime, '')) NOT LIKE 'image/%'
           OR LOWER(COALESCE(mime, '')) = 'image/svg+xml'
        """
    ).rowcount
    conn.commit()
    return deleted


def apply_mime_filter_to_manifest_db(db_path: str = IMAGE_DB_PATH) -> dict[str, int]:
    """Apply the image-only MIME filter directly to the manifest database."""
    conn = utils.connect_db(db_path)
    try:
        before = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        non_image = conn.execute(
            """
            SELECT COUNT(*)
            FROM images
            WHERE LOWER(COALESCE(mime, '')) NOT LIKE 'image/%'
               OR LOWER(COALESCE(mime, '')) = 'image/svg+xml'
            """
        ).fetchone()[0]
        deleted = purge_non_image_manifest_records(conn)
        after = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        return {"before": before, "non_image": non_image, "deleted": deleted, "after": after}
    finally:
        conn.close()


def upsert_image_records(
    conn: sqlite3.Connection,
    category: str,
    records: list[dict],
    source_scope: str = constants.CATEGORY_SOURCE_SCOPE_ROOT,
    parent_category: str | None = None,
) -> None:
    """Upsert image records while preserving future download status fields."""
    conn.execute(
        """
        INSERT INTO categories(category, parent_category, source_scope, fetched_at, fetch_status, error)
        VALUES (?, ?, ?, ?, ?, NULL)
        ON CONFLICT(category) DO UPDATE SET
            parent_category=COALESCE(excluded.parent_category, categories.parent_category),
            source_scope=excluded.source_scope,
            fetched_at=excluded.fetched_at,
            fetch_status=?,
            error=NULL
        """,
        (
            category,
            parent_category,
            source_scope,
            utc_now(),
            constants.FETCH_STATUS_OK,
            constants.FETCH_STATUS_OK,
        ),
    )
    for record in records:
        conn.execute(
            """
            INSERT INTO images(
                series, wiki_title, power_type, operator_en_json, root_category, category, category_path_json, file_title,
                pageid, image_url, thumb_url, mime, width, height, size, sha1, extmetadata_json,
                excluded, exclude_reason, fetched_at
            ) VALUES (
                :series, :wiki_title, :power_type, :operator_en_json, :root_category, :category, :category_path_json, :file_title,
                :pageid, :image_url, :thumb_url, :mime, :width, :height, :size, :sha1, :extmetadata_json,
                :excluded, :exclude_reason, :fetched_at
            )
            ON CONFLICT(series, category, file_title) DO UPDATE SET
                pageid=excluded.pageid,
                image_url=excluded.image_url,
                thumb_url=excluded.thumb_url,
                power_type=excluded.power_type,
                mime=excluded.mime,
                width=excluded.width,
                height=excluded.height,
                size=excluded.size,
                sha1=excluded.sha1,
                category_path_json=excluded.category_path_json,
                extmetadata_json=excluded.extmetadata_json,
                excluded=excluded.excluded,
                exclude_reason=excluded.exclude_reason,
                fetched_at=excluded.fetched_at
            """,
            record,
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO image_categories(file_title, category, source_scope)
            VALUES (?, ?, ?)
            """,
            (record["file_title"], category, source_scope),
        )
    conn.commit()


def get_category_fetch_status(conn: sqlite3.Connection, category: str) -> str | None:
    row = conn.execute(
        "SELECT fetch_status FROM categories WHERE category = ?",
        (category,),
    ).fetchone()
    return row[0] if row else None


def should_fetch_category_files(
    conn: sqlite3.Connection,
    category: str,
    reprocess: bool = False,
) -> bool:
    """Return whether the direct file manifest for one category should be crawled."""
    if reprocess:
        return True
    return get_category_fetch_status(conn, category) != constants.FETCH_STATUS_OK

# ================== Manifest Crawling 流程 ==================

def should_skip_category(row: pd.Series, category: str) -> str | None:
    """Return the exclusion reason for a category, if it should not be crawled."""
    return category_exclude_reason(row, category)


def collect_category_records(
    row: pd.Series,
    category: str,
    path: list[str],
    depth: int,
    max_depth: int,
    max_files_per_category: int,
    visited_paths: set[tuple[str, ...]],
) -> list[dict]:
    """Collect image records from a category and optionally recurse into subcategories."""
    path_key = tuple(path)
    if path_key in visited_paths:
        return []
    visited_paths.add(path_key)

    records = build_image_records(
        row, category, max_files=max_files_per_category, category_path=path
    )
    logger.info('depth=%d Category:%s -> %d 个文件', depth, category, len(records))

    if max_depth != -1 and depth >= max_depth:
        return records

    subcats = fetch_subcategories(category) or []
    for subcat in subcats:
        if should_skip_category(row, subcat):
            continue
        records.extend(
            collect_category_records(
                row=row,
                category=subcat,
                path=path + [subcat],
                depth=depth + 1,
                max_depth=max_depth,
                max_files_per_category=max_files_per_category,
                visited_paths=visited_paths,
            )
        )
    return records


def crawl_category_records_with_checkpoint(
    conn: sqlite3.Connection,
    row: pd.Series,
    category: str,
    path: list[str],
    depth: int,
    max_depth: int,
    max_files_per_category: int,
    visited_paths: set[tuple[str, ...]],
    reprocess: bool = False,
) -> list[dict]:
    """Crawl one category subtree, skipping already fetched category file manifests.

    The checkpoint is category-local: an ``ok`` category skips only direct file
    fetching for that category. Subcategories are still discovered and visited so
    a previous interrupted run can resume inside a partially processed subtree.
    """
    path_key = tuple(path)
    if path_key in visited_paths:
        return []
    visited_paths.add(path_key)

    records: list[dict] = []
    source_scope = (
        constants.CATEGORY_SOURCE_SCOPE_ROOT
        if len(path) == 1
        else constants.CATEGORY_SOURCE_SCOPE_RECURSIVE
    )
    parent_category = path[-2] if len(path) > 1 else None

    if should_fetch_category_files(conn, category, reprocess=reprocess):
        category_records = build_image_records(
            row,
            category,
            max_files=max_files_per_category,
            category_path=path,
        )
        upsert_image_records(
            conn,
            category,
            category_records,
            source_scope=source_scope,
            parent_category=parent_category,
        )
        records.extend(category_records)
        logger.info("depth=%d Category:%s -> %d 个文件", depth, category, len(category_records))
    else:
        logger.info("depth=%d Category:%s -> 已有 checkpoint，跳过直接文件 manifest", depth, category)

    if max_depth != -1 and depth >= max_depth:
        return records

    subcats = fetch_subcategories(category) or []
    for subcat in subcats:
        if should_skip_category(row, subcat):
            continue
        records.extend(
            crawl_category_records_with_checkpoint(
                conn=conn,
                row=row,
                category=subcat,
                path=path + [subcat],
                depth=depth + 1,
                max_depth=max_depth,
                max_files_per_category=max_files_per_category,
                visited_paths=visited_paths,
                reprocess=reprocess,
            )
        )
    return records


def crawl_root_manifest_sample(
    sample_series: list[str] | None = None,
    max_files_per_category: int = 40,
    max_depth: int = -1, # -1:不设上限
    db_path: str = IMAGE_DB_PATH,
    models: pd.DataFrame | None = None,
    reprocess: bool = False,
) -> pd.DataFrame:
    """Crawl manifest records for selected series; max_depth=0 means root only."""
    models = load_commons_models() if models is None else models
    if sample_series is None:
        sample = models.copy()
    else:
        sample = models[models["series"].isin(sample_series)].copy()
    sample = sample[sample["commons_root_category"].notna()]

    conn = utils.connect_db(db_path)
    all_records = []
    try:
        logger.info("准备爬取 manifest：%d 个车型", len(sample))
        for _, row in tqdm(sample.iterrows(), total=len(sample), desc="Manifest series", unit="series"):
            root_category = row["commons_root_category"]
            records = crawl_category_records_with_checkpoint(
                conn=conn,
                row=row,
                category=root_category,
                path=[root_category],
                depth=0,
                max_depth=max_depth,
                max_files_per_category=max_files_per_category,
                visited_paths=set(),
                reprocess=reprocess,
            )

            all_records.extend(records)
            logger.info('%s：Category:%s -> 本次新增/更新 %d 个文件', row["series"], root_category, len(records))

        purged = purge_non_image_manifest_records(conn)
        logger.info('MIME 过滤已从 manifest 数据库移除 %d 条非图片记录', purged)
    finally:
        conn.close()

    return pd.DataFrame(all_records)









# ================== pipeline主函数 ==================
def main(config_override=None):
    cfg = config_override or config
    utils.init_db(config=cfg)
    crawler_config = cfg["crawler"]
    logger.info("正在读取 Commons 车型映射 CSV")

    db_path = utils.join_data_root(cfg["path"]["db_path"], config=cfg)
    models = load_commons_models(utils.join_data_root(cfg["path"]["series_commons_path"], config=cfg))
    selected_models, scope_name = select_models_to_crawl(models, crawler_config)
    full_on = _as_bool(crawler_config["full_series_crawling"])
    if full_on:
        logger.warning("已设置为全量爬取，预计耗时较长；如非必要请先用 series_test_scope 测试部分车型")

    max_files_per_category = int(crawler_config["manifest_max_files_per_category"])
    max_depth = int(crawler_config["manifest_max_depth"])
    reprocess = _as_bool(crawler_config["manifest_reprocess"])
    logger.info(
        "Manifest 爬取模式：%s；车型数：%d；每个分类最多文件数：%d；递归深度：%d；覆盖重爬：%s",
        scope_name,
        len(selected_models),
        max_files_per_category,
        max_depth,
        reprocess,
    )

    if selected_models.empty:
        logger.warning("没有选中任何车型，跳过 manifest 爬取")
        return

    sample_manifest = crawl_root_manifest_sample(
        sample_series=None,
        max_files_per_category=max_files_per_category,
        max_depth=max_depth,
        db_path=db_path,
        models=selected_models,
        reprocess=reprocess,
    )
    logger.info("本次 manifest 爬取获得 %d 条原始记录", len(sample_manifest))

    filter_result = apply_mime_filter_to_manifest_db(db_path)
    logger.info(
        "MIME 过滤完成：过滤前 %d 条；非图片 %d 条；删除 %d 条；过滤后 %d 条",
        filter_result["before"],
        filter_result["non_image"],
        filter_result["deleted"],
        filter_result["after"],
    )
    
if __name__ == "__main__":
    main()
