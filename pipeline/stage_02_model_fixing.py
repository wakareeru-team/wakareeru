import ast
import json
import os
import random
import re
import time

import httpx
import pandas as pd

import constants
import utils


PROJECT_ROOT = utils.get_project_root()
logger = utils.get_logger("stage_02_model_fixing")

MANUAL_ACTIONS = {"set", "keep", "exclude"}
MANUAL_KEY_COLUMNS = {"series", "wiki_title"}
MANUAL_META_COLUMNS = MANUAL_KEY_COLUMNS | {"action", "reason", "comment"}
MANUAL_COLUMN_ALIASES = {
    "set_series": "series",
    "set_wiki_title": "wiki_title",
}
MANUAL_LITERAL_COLUMNS = {
    "operator_page_title",
    "operator_jp",
    "operator_en",
    "commons_operator_roots",
    "commons_candidates",
}
MANUAL_SOURCE_SERIES_COL = "__manual_source_series"
MANUAL_SOURCE_WIKI_TITLE_COL = "__manual_source_wiki_title"
COMMONS_COLUMNS = [
    "commons_prefix",
    "commons_root_category",
    "commons_root_decision",
    "commons_candidates",
    "needs_review",
    "commons_operator_roots",
]


def parse_json_list(value) -> list[str]:
    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    return json.loads(value)


def _wiki_base(wiki_title: str) -> str:
    return str(wiki_title).split("#", 1)[0]


def _manual_key(series: str, wiki_title: str) -> tuple[str, str]:
    return str(series), _wiki_base(wiki_title)


def _manual_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _parse_manual_literal(value: str):
    parsed = ast.literal_eval(value)
    if isinstance(parsed, tuple):
        return list(parsed)
    return parsed


def _coerce_manual_value(col: str, value: str):
    if col in MANUAL_LITERAL_COLUMNS:
        return _parse_manual_literal(value)
    if col == "needs_review":
        return value.lower() == "true"
    return value


def load_manual_overrides(path: str | os.PathLike | None) -> dict[tuple[str, str], dict]:
    if not path:
        return {}

    path = utils.join_root_path(path)
    if not os.path.exists(path):
        logger.warning("未找到人工修正文件：%s", path)
        return {}

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"series", "wiki_title", "action"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"人工修正文件缺少列：{sorted(missing)}")

    overrides = {}
    for _, row in df.iterrows():
        action = _manual_value(row["action"]).lower()
        if action not in MANUAL_ACTIONS:
            raise ValueError(f"{row['series']} 的人工修正 action 未知：{action}")

        key = _manual_key(_manual_value(row["series"]), _manual_value(row["wiki_title"]))
        if key in overrides:
            raise ValueError(f"人工修正 key 重复：{key}")

        values = {}
        for col in df.columns:
            value = _manual_value(row[col])
            if col in MANUAL_META_COLUMNS or not value:
                continue
            target_col = MANUAL_COLUMN_ALIASES.get(col, col)
            values[target_col] = _coerce_manual_value(target_col, value)

        overrides[key] = {"action": action, "values": values}

    return overrides


def _manual_for_row(row: pd.Series, manual_overrides: dict[tuple[str, str], dict]) -> dict | None:
    source_series = row.get(MANUAL_SOURCE_SERIES_COL)
    source_wiki_title = row.get(MANUAL_SOURCE_WIKI_TITLE_COL)
    if _manual_value(source_series) and _manual_value(source_wiki_title):
        override = manual_overrides.get(_manual_key(source_series, source_wiki_title))
        if override:
            return override
    return manual_overrides.get(_manual_key(row["series"], row["wiki_title"]))


def apply_manual_pre_root(df: pd.DataFrame, manual_overrides: dict[tuple[str, str], dict]) -> pd.DataFrame:
    if not manual_overrides:
        return df

    df = df.copy()
    keep_mask = []
    for idx, row in df.iterrows():
        override = _manual_for_row(row, manual_overrides)
        if override and override["action"] == "exclude":
            keep_mask.append(False)
            continue

        keep_mask.append(True)
        if override and override["action"] == "set":
            df.at[idx, MANUAL_SOURCE_SERIES_COL] = row["series"]
            df.at[idx, MANUAL_SOURCE_WIKI_TITLE_COL] = row["wiki_title"]
            for col, value in override["values"].items():
                if col in df.columns:
                    df.at[idx, col] = value

    return df.loc[keep_mask].reset_index(drop=True)


def apply_manual_output(df: pd.DataFrame, manual_overrides: dict[tuple[str, str], dict]) -> pd.DataFrame:
    if not manual_overrides:
        return df

    df = df.copy()
    for idx, row in df.iterrows():
        override = _manual_for_row(row, manual_overrides)
        if not override or override["action"] != "set":
            continue
        for col, value in override["values"].items():
            if col in df.columns:
                df.at[idx, col] = value
    return df


def _parse_manual_mapping(value) -> dict:
    parsed = value if isinstance(value, dict) else ast.literal_eval(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"人工映射必须是 dict：{value}")
    return parsed


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[MANUAL_SOURCE_SERIES_COL, MANUAL_SOURCE_WIKI_TITLE_COL],
        errors="ignore",
    )


def _row_key(row: pd.Series) -> tuple[str, str]:
    return _manual_key(row["series"], row["wiki_title"])


def _load_commons_cache(path: str | os.PathLike) -> dict[tuple[str, str], dict]:
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path, dtype=object, keep_default_na=False, encoding="utf-8")
    missing = {"series", "wiki_title"} - set(df.columns)
    if missing:
        logger.warning("Commons 缓存缺少 key 列 %s，已忽略缓存", sorted(missing))
        return {}

    commons_cols = [col for col in COMMONS_COLUMNS if col in df.columns]
    cache = {}
    for _, row in df.iterrows():
        values = {col: row[col] for col in commons_cols}
        if "needs_review" in values:
            values["needs_review"] = str(values["needs_review"]).strip().lower() == "true"
        cache[_row_key(row)] = values
    return cache


def _cached_commons_result(row: pd.Series, cache: dict[tuple[str, str], dict]) -> dict | None:
    cached = cache.get(_row_key(row))
    if not cached:
        return None
    return {col: cached.get(col) for col in COMMONS_COLUMNS}


def _empty_review_result(row: pd.Series) -> dict:
    return {
        "commons_prefix": "",
        "commons_root_category": None,
        "commons_root_decision": "增量模式未联网补查，且缺少 Commons 缓存",
        "commons_candidates": [],
        "needs_review": True,
        "commons_operator_roots": {},
    }


def _katakana_to_romaji(text: str) -> str:
    result, i = "", 0
    while i < len(text):
        two = text[i : i + 2]
        if two in constants._DIGRAPHS:
            result += constants._DIGRAPHS[two]
            i += 2
        elif text[i] in constants._SINGLE:
            result += constants._SINGLE[text[i]]
            i += 1
        else:
            result += text[i]
            i += 1
    return result


def _operator_prefixes(operator_jp: list[str], series_type: str, wiki_title: str) -> list[str]:
    if series_type == "新幹線電車":
        return ["Shinkansen"]
    if str(wiki_title).startswith("国鉄"):
        return ["JNR"]

    prefixes = []
    for op in operator_jp:
        prefix = constants.OPERATOR_PREFIX.get(op, op)
        if prefix not in prefixes:
            prefixes.append(prefix)
    return prefixes


def series_to_commons_prefixes(
    series: str,
    operator_jp: list[str],
    series_type: str,
    wiki_title: str,
    manual: dict | None = None,
) -> list[str]:
    if manual and manual["action"] == "set" and manual["values"].get("commons_prefix"):
        return [manual["values"]["commons_prefix"]]

    name = re.sub(r"[系形]$", "", series)
    m = re.match(r"^([ァ-ヿ]+)(.*)", name)

    prefixes = []
    if series_type == "蒸気機関車":
        prefixes.append(f"{name} steam locomotive")

    operator_prefixes = _operator_prefixes(operator_jp, series_type, wiki_title)
    for op in operator_prefixes:
        if m:
            romaji = _katakana_to_romaji(m.group(1)).capitalize()
            rest = m.group(2)
            candidates = [f"{op} {romaji} {rest}".strip()]
            if rest:
                candidates.append(f"{op} {rest}")
        else:
            candidates = [f"{op} {name}"]

        for prefix in candidates:
            if prefix not in prefixes:
                prefixes.append(prefix)

    return prefixes


def _commons_query(params: dict, max_retries: int = 3, base_sleep: float = 1.0) -> dict | None:
    for attempt in range(max_retries + 1):
        try:
            resp = httpx.get(
                constants.COMMONS_API_URL,
                params=params,
                headers=constants.COMMONS_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                error = data["error"]
                raise RuntimeError(f'{error.get("code", "api-error")} {error.get("info", data)}')
            return data
        except (httpx.HTTPError, ValueError, RuntimeError) as exc:
            if attempt >= max_retries:
                logger.error("Commons 请求失败：%s（%s）", params, exc)
                return None
            sleep_s = base_sleep * (2**attempt) + random.uniform(0, 0.5)
            logger.warning(
                "Commons 临时错误，%.1f 秒后重试（%d/%d）",
                sleep_s,
                attempt + 1,
                max_retries,
            )
            time.sleep(sleep_s)


def fetch_commons_categories(prefix: str, limit: int = 20) -> list[str] | None:
    data = _commons_query(
        {
            "action": "query",
            "list": "allcategories",
            "acprefix": prefix,
            "aclimit": limit,
            "format": "json",
        }
    )
    if data is None:
        return None
    return [r["*"] for r in data.get("query", {}).get("allcategories", [])]


def fetch_parent_categories(category: str) -> list[str] | None:
    params = {
        "action": "query",
        "titles": f"Category:{category}",
        "prop": "categories",
        "cllimit": "max",
        "format": "json",
    }
    parents = []
    while True:
        data = _commons_query(params)
        if data is None:
            return None
        for page in data.get("query", {}).get("pages", {}).values():
            for cat in page.get("categories", []):
                parents.append(cat["title"].removeprefix("Category:"))
        if "continue" not in data:
            return parents
        params.update(data["continue"])


def fetch_subcategories(category: str) -> list[str] | None:
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


def _dedupe(items: list[str]) -> list[str]:
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _promote_to_series(series_label: str, exact: str, series_cat: str) -> tuple[bool, str]:
    if not series_label.endswith("系"):
        return False, "精确匹配；当前行不是系"

    exact_parents = fetch_parent_categories(exact) or []
    if series_cat in exact_parents:
        return True, f'提升到系列分类："{series_cat}" 是 "{exact}" 的父分类'

    series_children = fetch_subcategories(series_cat) or []
    if exact in series_children:
        return True, f'提升到系列分类："{exact}" 是 "{series_cat}" 的子分类'

    return False, "精确匹配；未确认系列父子关系"


def choose_commons_root(series_label: str, prefix: str, candidates: list[str]) -> dict:
    exact = prefix if prefix in candidates else None
    series_cat = f"{prefix} series" if f"{prefix} series" in candidates else None
    plural = f"{prefix}s" if f"{prefix}s" in candidates else None

    if plural:
        return {"root": plural, "decision": "复数分类匹配", "needs_review": False}

    if exact and series_cat:
        promote, reason = _promote_to_series(series_label, exact, series_cat)
        root = series_cat if promote else exact
        return {"root": root, "decision": reason, "needs_review": False}

    if exact:
        return {"root": exact, "decision": "前缀精确匹配", "needs_review": False}

    if series_cat:
        return {"root": series_cat, "decision": "未命中精确分类，使用 series 分类", "needs_review": False}

    if candidates:
        return {"root": candidates[0], "decision": "回退到首个前缀搜索结果", "needs_review": True}

    return {"root": None, "decision": "没有 Commons 分类候选", "needs_review": True}


def _find_root_from_prefixes(series_label: str, prefixes: list[str], category_limit: int = 20) -> dict:
    all_candidates = []
    fallback = None

    for prefix in prefixes:
        candidates = fetch_commons_categories(prefix, limit=category_limit)
        if candidates is None:
            continue
        all_candidates.extend(candidates)
        if not candidates:
            continue

        chosen = choose_commons_root(series_label, prefix, candidates)
        result = {
            "commons_prefix": prefix,
            "commons_root_category": chosen["root"],
            "commons_root_decision": chosen["decision"],
            "commons_candidates": _dedupe(all_candidates),
            "needs_review": chosen["needs_review"],
        }
        if not chosen["needs_review"]:
            return result
        fallback = fallback or result

    if fallback:
        fallback["commons_candidates"] = _dedupe(all_candidates)
        return fallback

    return {
        "commons_prefix": prefixes[0] if prefixes else "",
        "commons_root_category": None,
        "commons_root_decision": "前缀未返回分类",
        "commons_candidates": _dedupe(all_candidates),
        "needs_review": True,
    }


def find_commons_root(
    row: pd.Series,
    manual_overrides: dict[tuple[str, str], dict] | None = None,
    category_limit: int = 20,
) -> dict:
    manual = _manual_for_row(row, manual_overrides or {})
    prefixes = series_to_commons_prefixes(
        row["series"],
        row["operator_jp"],
        row["type"],
        row["wiki_title"],
        manual=manual,
    )

    if manual and manual["action"] == "set" and manual["values"].get("commons_root_category"):
        root = manual["values"]["commons_root_category"]
        operator_roots = (
            _parse_manual_mapping(manual["values"]["commons_operator_roots"])
            if manual["values"].get("commons_operator_roots")
            else {op: root for op in row["operator_en"]}
        )
        return {
            "commons_prefix": prefixes[0] if prefixes else manual["values"].get("commons_prefix", ""),
            "commons_root_category": root,
            "commons_root_decision": "人工指定",
            "commons_operator_roots": operator_roots,
            "commons_candidates": [],
            "needs_review": manual["values"].get("needs_review", False),
        }

    root = _find_root_from_prefixes(row["series"], prefixes, category_limit=category_limit)
    operator_roots = {}
    for op_jp, op_en in zip(row["operator_jp"], row["operator_en"]):
        op_prefixes = series_to_commons_prefixes(
            row["series"],
            [op_jp],
            row["type"],
            row["wiki_title"],
            manual=None,
        )
        op_root = _find_root_from_prefixes(row["series"], op_prefixes, category_limit=category_limit)
        if op_root["commons_root_category"]:
            operator_roots[op_en] = op_root["commons_root_category"]

    root["commons_operator_roots"] = operator_roots
    if manual and manual["action"] == "keep":
        root["commons_root_decision"] = f'人工保留：{root["commons_root_decision"]}'
        root["needs_review"] = False
    return root


def main(config=None):
    utils.init_db()
    config = config or utils.load_pipeline_config()

    manual_overrides = load_manual_overrides(config["path"].get("manual_series_overrides_path"))
    series_commons_path = utils.join_root_path(config["path"]["series_commons_path"])
    commons_cache = _load_commons_cache(series_commons_path)
    all_model = pd.read_csv(utils.join_root_path(config["path"]["series_list_path"]), encoding="utf-8")
    for col in ["operator_page_title", "operator_jp", "operator_en"]:
        all_model[col] = all_model[col].apply(parse_json_list)

    before_manual = len(all_model)
    all_model = apply_manual_pre_root(all_model, manual_overrides)
    logger.info(
        "已读取人工修正规则：%d 条；排除行：%d 条",
        len(manual_overrides),
        before_manual - len(all_model),
    )
    logger.info("已读取 Commons 缓存：%d 行", len(commons_cache))

    root_rows = []
    fetched_count = 0
    cached_count = 0
    missing_cache_count = 0
    for _, row in all_model.iterrows():
        manual = _manual_for_row(row, manual_overrides)
        cached = _cached_commons_result(row, commons_cache)
        should_fetch = manual is not None or cached is None and not commons_cache

        if should_fetch:
            result = find_commons_root(row, manual_overrides=manual_overrides)
            fetched_count += 1
        elif cached is not None:
            result = cached
            cached_count += 1
        else:
            result = _empty_review_result(row)
            missing_cache_count += 1

        root_rows.append(result)
        status = "待确认" if result["needs_review"] else "通过"
        if should_fetch:
            logger.info(
                '%s %s (%s) -> "%s" [%s]',
                status,
                row["series"],
                ", ".join(row["operator_jp"]),
                result["commons_root_category"],
                result["commons_root_decision"],
            )

    commons_root_df = pd.DataFrame(root_rows)
    for col in commons_root_df.columns:
        all_model[col] = commons_root_df[col]

    all_model = apply_manual_output(all_model, manual_overrides)
    all_model = drop_internal_columns(all_model)

    review_count = int(all_model["needs_review"].sum())
    logger.info(
        "Commons 车型映射共 %d 行；联网验证 %d 行；复用缓存 %d 行；缺少缓存 %d 行；待人工确认 %d 行",
        len(all_model),
        fetched_count,
        cached_count,
        missing_cache_count,
        review_count,
    )

    os.makedirs(os.path.dirname(series_commons_path), exist_ok=True)
    all_model.to_csv(series_commons_path, index=False, encoding="utf-8")
    logger.info("Commons 车型映射已保存到 %s", config["path"]["series_commons_path"])

    review_path = utils.join_root_path(config["path"]["commons_review_path"])
    os.makedirs(os.path.dirname(review_path), exist_ok=True)
    review_df = all_model[all_model["needs_review"] == True]
    review_df.to_csv(review_path, index=False, encoding="utf-8")
    if not review_df.empty:
        logger.warning("仍有 %d 行需要人工确认，已导出到 %s", len(review_df), config["path"]["commons_review_path"])


if __name__ == "__main__":
    main()
