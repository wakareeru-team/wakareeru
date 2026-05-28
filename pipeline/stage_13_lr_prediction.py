import json
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import constants
import utils

import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lr_model import register_legacy_main_alias

logger = utils.get_logger("stage_13_lr_prediction")


def main(config: dict | None = None) -> None:
    if config is None:
        config = utils.load_pipeline_config()

    utils.init_db(config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    lr_prediction_config = config["lr_prediction"]
    lr_config = config["logistic_regression_filter"]
    model_dir = utils.join_data_root(config["path"]["model_dir"], config=config)
    model_name = lr_prediction_config["lr_model_path"]
    if model_name == "latest":
        model_pointer_path = model_dir / lr_config["model_pointer_path"]
        with open(model_pointer_path, "r") as f:
            model_name = f.read().strip()
    model_path = Path(model_name)
    if not model_path.is_absolute():
        model_path = model_dir / model_path
    register_legacy_main_alias()
    model = joblib.load(model_path)
    logger.info(f"已加载logistic regression模型: {model_name}")

    # 跳过人工审核过的crop
    with sqlite3.connect(db_path) as conn:
        query = "SELECT id AS crop_id FROM crops WHERE noise_review_label IS NULL"
        crops_to_process = pd.read_sql_query(query, conn)

    loss_round_dir = utils.get_current_loss_round_dir(config)
    prediction_path = loss_round_dir / lr_prediction_config["prediction_file_name"]
    existing_predictions = pd.DataFrame()
    if prediction_path.exists():
        existing_predictions = pd.read_csv(prediction_path)
        if not lr_prediction_config["reprocess"]:
            predicted_crop_ids = set(existing_predictions["crop_id"].astype(int))
            crops_to_process = crops_to_process[
                ~crops_to_process["crop_id"].astype(int).isin(predicted_crop_ids)
            ]

    if len(crops_to_process) == 0:
        logger.info("没有需要进行LR预测的crop，跳过此阶段")
        return constants.STAGE_PASS # type: ignore
    logger.info(f"需要进行LR预测的crop数量: {len(crops_to_process)}")
    
    # 加载需要预测的crop的feature
    features = pd.read_csv(loss_round_dir / config['loss_analysis']['loss_feature_file_name'])
    
    features_to_predict = features[features['crop_id'].isin(crops_to_process['crop_id'])]
    features_to_predict = features_to_predict.reset_index(drop=True)
    missing_feature_count = len(crops_to_process) - len(features_to_predict)
    if missing_feature_count:
        logger.warning("有%d条待预测crop缺少loss feature，已跳过。", missing_feature_count)
    if features_to_predict.empty:
        logger.info("没有具备完整loss feature的crop可供LR预测，跳过此阶段")
        return constants.STAGE_PASS  # type: ignore
    feature_columns = config['logistic_regression_filter']['feature_columns']
    X = features_to_predict[feature_columns].astype(float)
    prediction = model.predict(X)
    probability = model.predict_proba(X)[:, 1]

    positive_label = lr_config["noise_positive_label"][0]
    clean_label = lr_config["clean_label"]
    predicted_labels = [positive_label if pred else clean_label for pred in prediction]
    logger.info(f"LR预测完成，正样本比例: {sum(prediction)}/{len(prediction)}={sum(prediction)/len(prediction):.2%}")
    prediction_df = features_to_predict.copy()
    prediction_df["noise_predicted_prob"] = probability.astype(float)
    prediction_df["noise_predicted_label"] = predicted_labels
    prediction_df["noise_prediction_model"] = model_name
    prediction_df["predicted_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if not existing_predictions.empty and not lr_prediction_config["reprocess"]:
        prediction_df = pd.concat([existing_predictions, prediction_df], ignore_index=True)
        prediction_df = prediction_df.drop_duplicates("crop_id", keep="last")
    prediction_df.to_csv(prediction_path, index=False)
    logger.info("LR预测结果已保存至 %s", prediction_path)

    if not lr_prediction_config["sync_to_db"]:
        logger.info("lr_prediction.sync_to_db=false，跳过数据库同步。")
        return constants.STAGE_COMPLETED # type: ignore

    rows = [
        (float(prob), label, model_name, int(crop_id))
        for prob, label, crop_id in zip(
            probability,
            predicted_labels,
            features_to_predict["crop_id"],
            strict=True,
        )
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            UPDATE crops
            SET noise_predicted_prob = ?,
                noise_predicted_label = ?,
                noise_prediction_model = ?
            WHERE id = ?
            """,
            rows,
        )
        conn.commit()
    logger.info("已写入%d条LR预测结果。", len(rows))
if __name__ == "__main__":
    main()
