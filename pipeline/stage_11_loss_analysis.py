
import json
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import constants
import utils


logger = utils.get_logger("stage_11_loss_analysis")

# manifest加载helper

from pathlib import Path
from typing import Callable

import numpy as np
import yaml



def load_crop_manifest(
    db_path: Path,
    series: list[str] | None = None,
    power_type: str | None = None,
    crop_status: str | None = None,
    min_score: float | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    label_granularity = "submodel",
) -> pd.DataFrame:
    """从SQLite读取crop元数据，并拼上本地原图路径。

    结果DataFrame包含 ``label`` 列，内容由 ``label_granularity`` 决定：
    - ``"series"``            : 直接复制 series
    - ``"fine_grained_series"``: 优先用 fine_grained_series，NULL时回落到 series
    - ``"submodel"``          : 优先用 submodel，NULL时依次回落到 fine_grained_series、series
    """
    where = ["i.downloaded_path IS NOT NULL", "i.downloaded_path != ''"]
    params: list[object] = []

    if series:
        where.append("c.series IN (" + ",".join(["?"] * len(series)) + ")")
        params.extend(series)
    if power_type is not None:
        where.append("c.power_type = ?")
        params.append(power_type)
    if crop_status is not None:
        where.append("c.crop_status = ?")
        params.append(crop_status)
    if min_score is not None:
        where.append("COALESCE(c.detector_score, 0) >= ?")
        params.append(min_score)

    sql_limit = None if shuffle else limit
    limit_sql = "LIMIT ?" if sql_limit is not None else ""
    if sql_limit is not None:
        params.append(limit)

    sql = f"""
    SELECT
        c.id AS crop_id,
        c.image_id,
        c.crop_index,
        c.series,
        c.power_type,
        c.detector_label,
        c.detector_score,
        c.box_x1, c.box_y1, c.box_x2, c.box_y2,
        i.downloaded_path,
        i.file_title,
        i.width AS image_width,
        i.height AS image_height,
        i.fine_grained_series,
        i.submodel
    FROM crops c
    JOIN images i ON i.id = c.image_id
    WHERE {' AND '.join(where)}
    ORDER BY c.id
    {limit_sql}
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    if shuffle and len(df):
        n = min(limit or len(df), len(df))
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)

    # 计算有效分类标签
    if label_granularity == "submodel":
        df["label"] = df["submodel"].fillna(df["fine_grained_series"]).fillna(df["series"])
    elif label_granularity == "fine_grained_series":
        df["label"] = df["fine_grained_series"].fillna(df["series"])
    else:
        df["label"] = df["series"]

    return df



def main(config: dict | None = None) -> None:
    if config is None:
        config = utils.load_pipeline_config()
    noise_detection_cfg = config['noise_detection']
    utils.init_db(config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    df_crops = load_crop_manifest(db_path, label_granularity=noise_detection_cfg['label_granularity'])
    loss_noise_tracking_cfg = config['loss_noise_tracking']
    logger.info("开始提取损失分析数据")
    
    df_crops = load_crop_manifest(
    db_path=db_path,
    series= None if noise_detection_cfg.get("full_series") else noise_detection_cfg.get("series_test_scope"),
    label_granularity=noise_detection_cfg['label_granularity'],
    )
    loss_tracking_path = utils.get_current_loss_round_dir(config) / loss_noise_tracking_cfg["loss_history_file_name"]
    epoch_tracking_path = utils.get_current_loss_round_dir(config) / loss_noise_tracking_cfg["epoch_history_file_name"]
    lossdf = pd.read_csv(utils.join_data_root(loss_tracking_path))
    epochdf = pd.read_csv(utils.join_data_root(epoch_tracking_path))


    LABEL_COL = "label"
    labels = sorted(df_crops[LABEL_COL].dropna().astype(str).unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}


    lossdf['pred_label'] = lossdf['pred_id'].apply(lambda x: id_to_label[x])


    # 聚合crop 损失特征
    loss_feature = (
    lossdf.groupby("crop_id")["loss_value"]
    .agg(["mean", "std", "count"])
    .reset_index()
    )
    # 预测错误率
    error_rate = (
        lossdf.groupby("crop_id")["correct"]
        .agg(lambda x: 1.0 - float(np.mean(x)))
        .rename("error_rate")
        .reset_index()
    )


    # 尾部损失均值
    tail_loss = lossdf.groupby('crop_id')['loss_value'].agg(lambda x: np.mean(x.tail(5))).rename('loss_tail_mean').reset_index()


    # 模型该样本的标签，以众数为准，及其占比
    pred_summary = (
        lossdf.groupby("crop_id")
        .agg(
            pred_label=("pred_label", lambda x: x.value_counts().index[0]),
            pred_label_rate=("pred_label", lambda x: x.value_counts(normalize=True).iloc[0]),
        )
        .reset_index()
    )

    loss_feature = loss_feature.merge(error_rate, on="crop_id", how="left")
    loss_feature = loss_feature.merge(pred_summary, on="crop_id", how="left")
    loss_feature = loss_feature.merge(tail_loss, on="crop_id", how="left")
    crop_labels = df_crops[["crop_id", "label"]].drop_duplicates("crop_id")
    loss_feature = loss_feature.merge(crop_labels, on="crop_id", how="left")

    
    # inter-label loss quantile分析
    loss_feature['loss_mean_pct_in_label'] = loss_feature.groupby('label')['mean'].rank(pct=True)
    loss_feature.head(10)
    loss_feature_path = utils.get_current_loss_round_dir(config) / config['loss_analysis']['loss_feature_file_name']
    loss_feature.to_csv(utils.join_data_root(loss_feature_path), index=False)
    logger.info("已提取%d条损失分析数据，包含loss趋势，尾部均值，模型预测和inter label percentile.", len(loss_feature))
    logger.info("损失分析数据提取完成，保存至 %s", loss_feature_path)


    # v1最简评分
    loss_feature["noise_score_v1"] = (
        loss_feature["loss_mean_pct_in_label"] +
        loss_feature["error_rate"]
    )
    
    # 保存到数据库
    
    scores = loss_feature[["noise_score_v1","crop_id"]].copy().to_dict(orient="records")
    scores = [tuple(row.values()) for row in scores]
    len(scores)
    utils.init_db()  # 确保数据库已同步
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.executemany('''
                        UPDATE crops
                        SET noise_score_v1 = ?
                        WHERE id = ?''', scores)
        conn.commit()
    logger.info('已将冷启动简单噪声分数更新到数据库的 crops 表中。')
    if config['loss_analysis']['request_manual_review']:
        logger.info("已完成损失分析数据提取，并更新了简单噪声分数。请使用 review 工具对高分样本进行人工审核，完成后再次运行管线以继续后续阶段。")
        return constants.STAGE_INTERRUPT  # type: ignore # 请求人工审核，管线中断，等待后续手动触发
if __name__ == "__main__":
    main()
