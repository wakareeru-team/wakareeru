import json
import re
import sqlite3

import pandas as pd

import constants
import utils

logger = utils.get_logger("stage_13b_label_metadata_translation")

TRANSLATION_COLUMNS = [
    "label_ja",
    "label_en",
    "label_zh",
    "operator_ja_json",
    "operator_en_json",
    "operator_zh_json",
    "wiki_title_ja",
    "source_series",
    "note",
]
JAPANESE_TEXT_RE = re.compile(r"[ぁ-んァ-ヶ一-龠々]")
JAPANESE_KANA_RE = re.compile(r"[ぁ-んァ-ヶ]")


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _most_common_text(values: pd.Series) -> str:
    cleaned = values.map(_clean_text)
    cleaned = cleaned[cleaned.ne("")]
    if cleaned.empty:
        return ""
    counts = cleaned.value_counts()
    return sorted(counts[counts.eq(counts.max())].index)[0]


def _parse_json_array(value: object, *, label_ja: str, column: str) -> list[str]:
    try:
        parsed = json.loads(_clean_text(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label_ja!r} 的 {column} 不是合法JSON") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"{label_ja!r} 的 {column} 必须是字符串数组")
    values = [item.strip() for item in parsed]
    if any(not item for item in values):
        raise ValueError(f"{label_ja!r} 的 {column} 包含空字符串")
    return values


def _contains_link(value: str) -> bool:
    return "http://" in value or "https://" in value or bool(re.search(r"\]\([^)]*\)", value))


def validate_translation_row(row: pd.Series) -> tuple:
    label_ja = _clean_text(row["label_ja"])
    label_en = _clean_text(row["label_en"])
    label_zh = _clean_text(row["label_zh"])
    wiki_title_ja = _clean_text(row["wiki_title_ja"])
    note = _clean_text(row["note"])
    if not label_ja or not label_en or not label_zh:
        raise ValueError(f"{label_ja or '(空label)'} 的label_en/label_zh尚未填写")

    operators = {
        language: _parse_json_array(
            row[f"operator_{language}_json"],
            label_ja=label_ja,
            column=f"operator_{language}_json",
        )
        for language in ("ja", "en", "zh")
    }
    if len({len(values) for values in operators.values()}) != 1:
        raise ValueError(f"{label_ja!r} 的三语operator数组长度不一致")
    if len(operators["ja"]) != len(set(operators["ja"])):
        raise ValueError(f"{label_ja!r} 包含重复operator_ja")

    all_text = [label_en, label_zh, wiki_title_ja, note]
    all_text.extend(value for values in operators.values() for value in values)
    if any(_contains_link(value) for value in all_text):
        raise ValueError(f"{label_ja!r} 包含链接污染")
    if JAPANESE_TEXT_RE.search(label_en):
        raise ValueError(f"{label_ja!r} 的label_en包含日文污染")
    if JAPANESE_KANA_RE.search(label_zh):
        raise ValueError(f"{label_ja!r} 的label_zh包含日文假名污染")
    if any("/" in value for value in operators["ja"]):
        raise ValueError(f"{label_ja!r} 的operator_ja包含双语言分隔符")
    if any(JAPANESE_TEXT_RE.search(value) for value in operators["en"]):
        raise ValueError(f"{label_ja!r} 的operator_en包含日文污染")
    if any(JAPANESE_KANA_RE.search(value) for value in operators["zh"]):
        raise ValueError(f"{label_ja!r} 的operator_zh包含日文假名污染")

    return (
        label_ja,
        label_en,
        label_zh,
        json.dumps(operators["ja"], ensure_ascii=False),
        json.dumps(operators["en"], ensure_ascii=False),
        json.dumps(operators["zh"], ensure_ascii=False),
        wiki_title_ja,
        note,
    )


def load_label_candidates(conn: sqlite3.Connection, label_column: str) -> pd.DataFrame:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label_column):
        raise ValueError(f"非法crops_storage.label_column: {label_column!r}")
    image_columns = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
    if label_column not in image_columns:
        raise ValueError(f"images不存在label列: {label_column!r}")

    rows = pd.read_sql_query(
        f"""
        SELECT
            COALESCE(
                NULLIF(TRIM(c.manual_corrected_label), ''),
                NULLIF(TRIM(i.{label_column}), ''),
                i.series
            ) AS label_ja,
            i.series AS source_series,
            i.wiki_title AS wiki_title_ja
        FROM crops c
        JOIN images i ON i.id = c.image_id
        """,
        conn,
    )
    rows["label_ja"] = rows["label_ja"].map(_clean_text)
    rows = rows[rows["label_ja"].ne("")].copy()
    return rows


def _canonical_operator_lookup(conn: sqlite3.Connection) -> dict[str, dict[str, list[str]]]:
    metadata = pd.read_sql_query(
        """
        SELECT label_ja, operator_ja_json, operator_en_json, operator_zh_json
        FROM label_metadata
        """,
        conn,
    )
    lookup = {}
    for row in metadata.itertuples(index=False):
        lookup[row.label_ja] = {
            language: _parse_json_array(
                getattr(row, f"operator_{language}_json"),
                label_ja=row.label_ja,
                column=f"operator_{language}_json",
            )
            for language in ("ja", "en", "zh")
        }
    return lookup


def build_translation_queue(
    candidates: pd.DataFrame,
    missing_labels: set[str],
    operator_lookup: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    queue_rows = []
    for label_ja in sorted(missing_labels):
        label_rows = candidates[candidates["label_ja"].eq(label_ja)]
        source_series = sorted(
            value for value in label_rows["source_series"].map(_clean_text).unique() if value
        )
        inherited = {"ja": [], "en": [], "zh": []}
        seen_ja: set[str] = set()
        for series in source_series:
            operators = operator_lookup.get(series)
            if operators is None:
                continue
            for index, operator_ja in enumerate(operators["ja"]):
                if operator_ja in seen_ja:
                    continue
                seen_ja.add(operator_ja)
                for language in ("ja", "en", "zh"):
                    inherited[language].append(operators[language][index])
        queue_rows.append(
            {
                "label_ja": label_ja,
                "label_en": "",
                "label_zh": "",
                "operator_ja_json": json.dumps(inherited["ja"], ensure_ascii=False),
                "operator_en_json": json.dumps(inherited["en"], ensure_ascii=False),
                "operator_zh_json": json.dumps(inherited["zh"], ensure_ascii=False),
                "wiki_title_ja": _most_common_text(label_rows["wiki_title_ja"]),
                "source_series": json.dumps(source_series, ensure_ascii=False),
                "note": "",
            }
        )
    return pd.DataFrame(queue_rows, columns=TRANSLATION_COLUMNS)


def main(config: dict | None = None) -> int:
    if config is None:
        config = utils.load_pipeline_config()

    utils.init_db(config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    translation_config = config["label_metadata_translation"]
    review_path = utils.join_data_root(translation_config["review_file_path"], config=config)
    label_column = config["crops_storage"]["label_column"]

    with sqlite3.connect(db_path) as conn:
        candidates = load_label_candidates(conn, label_column)
        existing_labels = {
            row[0] for row in conn.execute("SELECT label_ja FROM label_metadata")
        }
        missing_labels = set(candidates["label_ja"]) - existing_labels
        operator_lookup = _canonical_operator_lookup(conn)

        existing_review = pd.DataFrame(columns=TRANSLATION_COLUMNS)
        if review_path.exists():
            existing_review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
            missing_columns = set(TRANSLATION_COLUMNS) - set(existing_review.columns)
            if missing_columns:
                raise ValueError(f"翻译表缺少列: {sorted(missing_columns)}")
            if existing_review["label_ja"].duplicated().any():
                duplicates = sorted(
                    existing_review.loc[
                        existing_review["label_ja"].duplicated(keep=False), "label_ja"
                    ].unique()
                )
                raise ValueError(f"翻译表包含重复label_ja: {duplicates}")
            existing_review = existing_review.set_index("label_ja", drop=False)

        import_rows = []
        for label_ja in sorted(missing_labels & set(existing_review.index)):
            row = existing_review.loc[label_ja]
            if not _clean_text(row["label_en"]) or not _clean_text(row["label_zh"]):
                continue
            import_rows.append(validate_translation_row(row))

        if import_rows:
            conn.executemany(
                """
                INSERT INTO label_metadata (
                    label_ja, label_en, label_zh,
                    operator_ja_json, operator_en_json, operator_zh_json,
                    wiki_title_ja, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(label_ja) DO NOTHING
                """,
                import_rows,
            )
            conn.commit()
            imported_labels = {row[0] for row in import_rows}
            missing_labels -= imported_labels
            operator_lookup = _canonical_operator_lookup(conn)
            logger.info("已将%d条完整翻译写入label_metadata。", len(import_rows))

        fresh_queue = build_translation_queue(candidates, missing_labels, operator_lookup)
        if not existing_review.empty and not fresh_queue.empty:
            for index, row in fresh_queue.iterrows():
                label_ja = row["label_ja"]
                if label_ja not in existing_review.index:
                    continue
                for column in TRANSLATION_COLUMNS:
                    fresh_queue.at[index, column] = existing_review.loc[label_ja, column]

    review_path.parent.mkdir(parents=True, exist_ok=True)
    fresh_queue.to_csv(review_path, index=False, encoding="utf-8")
    if not fresh_queue.empty:
        logger.warning(
            "仍有%d条新label待翻译，已导出至%s；填写后重跑本阶段。",
            len(fresh_queue),
            review_path,
        )
        return constants.STAGE_INTERRUPT

    logger.info("当前所有label均已有规范翻译；翻译队列为空。")
    return constants.STAGE_COMPLETED


if __name__ == "__main__":
    main()
