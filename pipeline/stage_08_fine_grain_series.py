import io
import sqlite3
from pathlib import Path

import pandas as pd

import utils

logger = utils.get_logger("stage_08_fine_grain_series")

FINE_GRAINED_SERIES_MATCHING_COLS = [
    "submodel",
    "bandai",
    "special_formation",
    "special_livery",
    "operator_en",
]
FINE_GRAINED_SERIES_RULE_COLS = [
    "series",
    *FINE_GRAINED_SERIES_MATCHING_COLS,
    "fine_grained_series",
]
DEFAULT_RULES_PATH = "config/migrations/manual_fine_grained_series.csv"


def load_fine_grained_rules(rules_path: str | Path) -> list[dict]:
    """Read the manual fine-grained split CSV, tolerating trailing commas."""
    with open(rules_path, "r", encoding="utf-8") as f:
        content = f.read()
    cleaned = "\n".join(line.rstrip(",") for line in content.splitlines())
    df = pd.read_csv(io.StringIO(cleaned), dtype=str, keep_default_na=False)
    df = df[[c for c in FINE_GRAINED_SERIES_RULE_COLS if c in df.columns]].apply(
        lambda col: col.str.strip()
    )
    return df.to_dict("records")


def rule_summary(rule: dict) -> str:
    parts = []
    if rule.get("series"):
        parts.append(f"series={rule['series']}")
    for col in FINE_GRAINED_SERIES_MATCHING_COLS:
        if rule.get(col):
            parts.append(f"{col}~'{rule[col]}'")
    return ", ".join(parts) or "(no conditions)"


def matches_rule(img: dict, rule: dict) -> bool:
    """series: exact; other non-empty fields: case-insensitive substring."""
    rule_series = rule.get("series", "")
    if rule_series and img.get("series", "") != rule_series:
        return False
    for col in FINE_GRAINED_SERIES_MATCHING_COLS:
        rule_val = rule.get(col, "")
        if not rule_val:
            continue
        if rule_val.lower() not in (img.get(col) or "").lower():
            return False
    return True


def apply_fine_grained_labels(conn: sqlite3.Connection, rules_path: str | Path) -> None:
    rules = load_fine_grained_rules(rules_path)
    if not rules:
        logger.warning("fine_grained_labels: no rules loaded from %s", rules_path)
        return

    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    if "fine_grained_series" not in existing_cols:
        conn.execute("ALTER TABLE images ADD COLUMN fine_grained_series TEXT")

    fetch_cols = ["id", "series"] + FINE_GRAINED_SERIES_MATCHING_COLS
    images_df = pd.read_sql_query(
        f"SELECT {', '.join(fetch_cols)} FROM images",
        conn,
        dtype=str,
    ).fillna("")

    rule_counts = [0] * len(rules)
    rule_match_counts = [0] * len(rules)
    updates: list[tuple[str, int]] = []

    for _, img in images_df.iterrows():
        img_dict = img.to_dict()
        label = img_dict["series"]
        rule_applied = False
        for i, rule in enumerate(rules):
            if matches_rule(img_dict, rule):
                rule_match_counts[i] += 1
                if not rule_applied:
                    label = rule["fine_grained_series"]
                    rule_counts[i] += 1
                    rule_applied = True
        updates.append((label, int(img_dict["id"])))

    conn.executemany("UPDATE images SET fine_grained_series = ? WHERE id = ?", updates)
    conn.commit()

    matched = sum(rule_counts)
    for i, rule in enumerate(rules):
        logger.info(
            "  rule %d [%s] -> '%s': %d images",
            i,
            rule_summary(rule),
            rule.get("fine_grained_series", ""),
            rule_counts[i],
        )
    unmatched_rule_indices = [i for i, count in enumerate(rule_match_counts) if count == 0]
    if unmatched_rule_indices:
        logger.warning(
            "fine_grained_series: %d rules matched no images:",
            len(unmatched_rule_indices),
        )
        for i in unmatched_rule_indices:
            rule = rules[i]
            logger.warning(
                "  rule %d [%s] -> '%s'",
                i,
                rule_summary(rule),
                rule.get("fine_grained_series", ""),
            )
    logger.info(
        "fine_grained_series: %d matched rules, %d defaulted to series (%d total)",
        matched,
        len(updates) - matched,
        len(updates),
    )


def main(config: dict | None = None) -> None:
    if config is None:
        config = utils.load_pipeline_config()

    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    rules_path = utils.join_project_root(
        config.get("fine_grain_series", {}).get("rules_path", DEFAULT_RULES_PATH)
    )

    logger.info("已对以下车型应用人工细分类别： %s", rules_path)
    with sqlite3.connect(db_path) as conn:
        apply_fine_grained_labels(conn, rules_path)


if __name__ == "__main__":
    main()
