# 哈哈最简单的一个
# LLm处理然后解析详细车型到数据库
import csv
import os
import sqlite3
from collections import Counter
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

if __name__ == "__main__":
    main()