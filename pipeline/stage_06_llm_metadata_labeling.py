# 哈哈最简单的一个
# LLm处理然后解析详细车型到数据库
import csv
import io
import os
import sqlite3
from collections import Counter
from pathlib import Path
from tqdm.auto import tqdm
from itertools import batched
import json
import constants
import utils
import pandas as pd
import time
from openai import OpenAI

config = utils.load_pipeline_config()
PROJECT_ROOT = utils.get_project_root()
logger = utils.get_logger("stage_06_llm_metadata_labeling")
IMAGE_DB_PATH = utils.join_root_path(config["path"]["db_path"])

BATCH_SIZE_DETAILS = 5
MAX_DETAILS_RETRIES = 3




def parse_llm_json_array(text: str) -> list[dict]:
    """Parse model output, tolerating markdown/prose around the JSON array."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("LLM output is not a JSON array")
    return data


def request_details_batch(openai_client, openai_model_name, system_prompt, batch_dict: list[dict]) -> list[dict]:
    """Analyze one category_path batch; retry when JSON is invalid or count mismatches."""
    batch_str = json.dumps(batch_dict, ensure_ascii=False)
    last_error = None
    for attempt in range(1, MAX_DETAILS_RETRIES + 1):
        response = openai_client.responses.create(
            model=openai_model_name,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batch_str},
            ],
            reasoning={"effort": "low"},
        )
        try:
            details = parse_llm_json_array(response.output_text)
            if len(details) != len(batch_dict):
                raise ValueError(f"Expected {len(batch_dict)} rows, got {len(details)}")
            return details
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            print(f"details JSON failed ({attempt}/{MAX_DETAILS_RETRIES}): {exc}")
            print((response.output_text or "")[:500])
            time.sleep(1)
    raise RuntimeError(f"Details LLM failed after retries: {last_error}")



# ========= Fine-Grained细化标签的写入 ==============

_FG_MATCH_COLS = constants.FINE_GRAINED_SERIES_MATCHING_COLS
_FG_RULE_COLS  = constants.FINE_GRAINED_SERIES_RULED_COLS


def _load_fine_grained_rules(rules_path: str | Path) -> list[dict]:
    """Read the manual fine-grained split CSV, tolerating trailing commas."""
    with open(rules_path, "r", encoding="utf-8") as f:
        content = f.read()
    cleaned = "\n".join(line.rstrip(",") for line in content.splitlines())
    df = pd.read_csv(io.StringIO(cleaned), dtype=str, keep_default_na=False)
    df = df[[c for c in _FG_RULE_COLS if c in df.columns]].apply(lambda col: col.str.strip())
    return df.to_dict("records")


def _rule_summary(rule: dict) -> str:
    parts = []
    if rule.get("series"):
        parts.append(f"series={rule['series']}")
    for col in _FG_MATCH_COLS:
        if rule.get(col):
            parts.append(f"{col}~'{rule[col]}'")
    return ", ".join(parts) or "(no conditions)"


def _matches_rule(img: dict, rule: dict) -> bool:
    """series: exact; other non-empty fields: case-insensitive substring."""
    rule_series = rule.get("series", "")
    if rule_series and img.get("series", "") != rule_series:
        return False
    for col in _FG_MATCH_COLS:
        rule_val = rule.get(col, "")
        if not rule_val:
            continue
        if rule_val.lower() not in (img.get(col) or "").lower():
            return False
    return True


def apply_fine_grained_labels(conn: sqlite3.Connection, rules_path: str | Path) -> None:
    rules = _load_fine_grained_rules(rules_path)
    if not rules:
        logger.warning("fine_grained_labels: no rules loaded from %s", rules_path)
        return

    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    if "fine_grained_series" not in existing_cols:
        conn.execute("ALTER TABLE images ADD COLUMN fine_grained_series TEXT")

    fetch_cols = ["id", "series"] + _FG_MATCH_COLS
    images_df = pd.read_sql_query(
        f"SELECT {', '.join(fetch_cols)} FROM images",
        conn,
        dtype=str,
    ).fillna("")

    rule_counts = [0] * len(rules)
    updates: list[tuple[str, int]] = []

    for _, img in images_df.iterrows():
        img_dict = img.to_dict()
        label = img_dict["series"]
        for i, rule in enumerate(rules):
            if _matches_rule(img_dict, rule):
                label = rule["fine_grained_series"]
                rule_counts[i] += 1
                break
        updates.append((label, int(img_dict["id"])))

    conn.executemany("UPDATE images SET fine_grained_series = ? WHERE id = ?", updates)
    conn.commit()

    matched = sum(rule_counts)
    for i, rule in enumerate(rules):
        logger.info("  rule %d [%s] -> '%s': %d images",
                    i, _rule_summary(rule), rule.get("fine_grained_series", ""), rule_counts[i])
    logger.info("fine_grained_series: %d matched rules, %d defaulted to series (%d total)",
                matched, len(updates) - matched, len(updates))












def main(config: dict | None = None):
    if config:
        config = config
    else:
        config = utils.load_pipeline_config()
        OPENAI_MODEL_NAME = config["llm_labeling"]["openai_model_name"]
    if os.environ.get("OPENAI_API_KEY") is None:
        logger.error("没有设置 OPENAI_API_KEY 环境变量，无法继续执行 LLM 相关的步骤。请设置环境变量后重试。")
        return
    
    openai_client = OpenAI()

    with sqlite3.connect(IMAGE_DB_PATH) as conn:
        category_paths = pd.read_sql_query("""
                                    SELECT category_path_json
                                    FROM images
                                    GROUP BY category_path_json
                                    """, conn)
    logger.info(f"共发现 {len(category_paths)} 条唯一的 category_path，准备进行LLM解析...")
    
    llm_details_rows = []
    for batch in batched(category_paths.itertuples(index=False), BATCH_SIZE_DETAILS):
        # Send parsed category_path lists, not raw JSON strings, so the prompt is cleaner.
        batch_dict = [
            {"category_path": json.loads(row.category_path_json)}
            for row in batch
        ]
        batch_details = request_details_batch(openai_client, OPENAI_MODEL_NAME, constants.LLM_LABEL_DETAIL_PROMPT, batch_dict)

        for row, detail in zip(batch, batch_details):
            detail["category_path_json"] = row.category_path_json
            detail["category_path"] = json.loads(row.category_path_json)
            llm_details_rows.append(detail)

        logger.info(f"已处理: {len(llm_details_rows)}/{len(category_paths)} category paths.")


    llm_details = pd.DataFrame(llm_details_rows)
    
    llm_details.to_csv(os.path.join(PROJECT_ROOT, "data", "llm_category_details.csv"), index=False)

    
        # 将 category details 回写到 images 表。幂等。
    DETAIL_COLS = ["submodel", "bandai", "operator_en", "operator_jp", "special_formation", "special_livery"]

    llm_details = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "llm_category_details.csv"), dtype={"bandai": str})

    def sql_null(v):
        """Coerce empty/sentinel strings to None so SQLite stores NULL."""
        if pd.isna(v):
            return None
        v = str(v).strip()
        return None if v.lower() in {"", "nan", "none", "null"} else v

    # Only normalize the text detail columns — category_path holds lists and must not be touched.
    llm_details[DETAIL_COLS] = llm_details[DETAIL_COLS].map(sql_null)

    missing = [c for c in ["category_path_json", *DETAIL_COLS] if c not in llm_details.columns]
    if missing:
        raise ValueError(f"details missing columns: {missing}")

    with sqlite3.connect(IMAGE_DB_PATH) as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
        for col in DETAIL_COLS:
            if col not in existing:
                conn.execute(f"ALTER TABLE images ADD COLUMN {col} TEXT")

        set_clause = ", ".join(f"{col} = ?" for col in DETAIL_COLS)
        update_rows = llm_details[DETAIL_COLS + ["category_path_json"]].values.tolist()
        conn.executemany(f"UPDATE images SET {set_clause} WHERE category_path_json = ?", update_rows)
        conn.commit()

        logger.info(f"已写入的 category details: {len(update_rows)}")

    rules_path = utils.join_root_path(config.get("llm_labeling", {}).get(
        "fine_grained_rules_path", "config/migrations/manual_fine_grained_series.csv"
    ))
    logger.info("Applying fine-grained series splits from %s", rules_path)
    with sqlite3.connect(IMAGE_DB_PATH) as conn:
        apply_fine_grained_labels(conn, rules_path)




if __name__ == "__main__":
    main()