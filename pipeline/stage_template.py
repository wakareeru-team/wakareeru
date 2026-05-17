"""Template for adding a new Wakareeru pipeline stage.

Copy this file to ``stage_XX_name.py`` and keep only the imports you use.
"""

import json
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import constants
import utils


logger = utils.get_logger("stage_template")


def main(config: dict | None = None) -> None:
    if config is None:
        config = utils.load_pipeline_config()

    utils.init_db(config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)

    logger.info("Stage template started. DB path: %s", db_path)


if __name__ == "__main__":
    main()
