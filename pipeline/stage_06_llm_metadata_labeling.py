import os
import sqlite3
from itertools import batched
import json
import constants
import utils
import pandas as pd
import time
from openai import OpenAI
from tqdm.auto import tqdm

config = utils.load_pipeline_config()
logger = utils.get_logger("stage_06_llm_metadata_labeling")

def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def build_web_search_tools(llm_config: dict) -> list[dict]:
    if not _as_bool(llm_config["web_search_enabled"]):
        return []

    tool = {"type": "web_search"}
    context_size = llm_config["web_search_context_size"]
    if context_size:
        tool["search_context_size"] = str(context_size)
    return [tool]




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


def request_details_batch(
    openai_client,
    openai_model_name,
    system_prompt,
    effort,
    batch_dict: list[dict],
    max_retries: int,
    tools: list[dict] | None = None,
    tool_choice: str | None = None,
) -> list[dict]:
    """Analyze one category_path batch; retry when JSON is invalid or count mismatches."""
    batch_str = json.dumps(batch_dict, ensure_ascii=False)
    last_error = None
    for attempt in range(1, max_retries + 1):
        request_kwargs = {
            "model": openai_model_name,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batch_str},
            ],
            "reasoning": {"effort": effort},
        }
        if tools:
            request_kwargs["tools"] = tools
            if tool_choice:
                request_kwargs["tool_choice"] = tool_choice

        response = openai_client.responses.create(**request_kwargs)
        try:
            details = parse_llm_json_array(response.output_text)
            if len(details) != len(batch_dict):
                raise ValueError(f"Expected {len(batch_dict)} rows, got {len(details)}")
            return details
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            print(f"details JSON failed ({attempt}/{max_retries}): {exc}")
            print((response.output_text or "")[:500])
            time.sleep(1)
    raise RuntimeError(f"Details LLM failed after retries: {last_error}")



def main(config: dict | None = None):
    if config is None:
        config = utils.load_pipeline_config()
    utils.init_db(config=config)
    llm_config = config["llm_labeling"]
    OPENAI_MODEL_NAME = llm_config["openai_model_name"]
    effort = llm_config['reasoning_effort']
    batch_size = int(llm_config["batch_size"])
    max_retries = int(llm_config["max_retries"])
    if os.environ.get("OPENAI_API_KEY") is None:
        logger.error("没有设置 OPENAI_API_KEY 环境变量，无法继续执行 LLM 相关的步骤。请设置环境变量后重试。")
        return
    
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    openai_client = OpenAI()
    tools = build_web_search_tools(llm_config)
    tool_choice = llm_config["web_search_tool_choice"] if tools else None
    if tools:
        logger.info("LLM metadata labeling 已启用 OpenAI web_search tool: %s", tools)

    with sqlite3.connect(db_path) as conn:
        category_paths = pd.read_sql_query("""
                                    SELECT category_path_json
                                    FROM images
                                    GROUP BY category_path_json
                                    """, conn)
    logger.info(f"共发现 {len(category_paths)} 条唯一的 category_path，准备进行LLM解析...")
    
    llm_details_rows = []
    with tqdm(total=len(category_paths), desc="LLM metadata labeling", unit="path") as pbar:
        for batch in batched(category_paths.itertuples(index=False), batch_size):
            # Send parsed category_path lists, not raw JSON strings, so the prompt is cleaner.
            batch_dict = [
                {"category_path": json.loads(row.category_path_json)}
                for row in batch
            ]
            batch_details = request_details_batch(
                openai_client,
                OPENAI_MODEL_NAME,
                constants.LLM_LABEL_DETAIL_PROMPT,
                effort,
                batch_dict,
                max_retries=max_retries,
                tools=tools,
                tool_choice=tool_choice,
            )

            for row, detail in zip(batch, batch_details):
                detail["category_path_json"] = row.category_path_json
                detail["category_path"] = json.loads(row.category_path_json)
                llm_details_rows.append(detail)

            pbar.update(len(batch_details))
            logger.info(f"已处理: {len(llm_details_rows)}/{len(category_paths)} category paths.")


    llm_details = pd.DataFrame(llm_details_rows)
    
    details_path = utils.join_data_root(
        config["path"]["llm_category_details_path"],
        config=config,
    )
    details_path.parent.mkdir(parents=True, exist_ok=True)
    llm_details.to_csv(details_path, index=False)

    
        # 将 category details 回写到 images 表。幂等。
    DETAIL_COLS = ["submodel", "bandai", "operator_en", "operator_jp", "special_formation", "special_livery"]

    llm_details = pd.read_csv(details_path, dtype={"bandai": str})

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

    with sqlite3.connect(db_path) as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
        for col in DETAIL_COLS:
            if col not in existing:
                conn.execute(f"ALTER TABLE images ADD COLUMN {col} TEXT")

        set_clause = ", ".join(f"{col} = ?" for col in DETAIL_COLS)
        update_rows = llm_details[DETAIL_COLS + ["category_path_json"]].values.tolist()
        conn.executemany(f"UPDATE images SET {set_clause} WHERE category_path_json = ?", update_rows)
        conn.commit()

        logger.info(f"已写入的 category details: {len(update_rows)}")


if __name__ == "__main__":
    main()
