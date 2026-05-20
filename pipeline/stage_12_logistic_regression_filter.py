"""Template for adding a new Wakareeru pipeline stage.

Copy this file to ``stage_XX_name.py`` and keep only the imports you use.
"""

import os
import re
import sqlite3
import sys
from pathlib import Path
import time
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

import constants
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
import joblib
from lr_model import LogisticRegressionWithThreshold

logger = utils.get_logger("stage_12_logistic_regression_filter")
    
def main(config: dict | None = None) -> None:
    if config is None:
        config = utils.load_pipeline_config()
    lr_config = config['logistic_regression_filter']
    utils.init_db(config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    logger.info("对人工标记数据进行logistic regression")
    features = pd.read_csv(utils.join_data_root(config['loss_analysis']['loss_feature_dir'], config=config))


    with sqlite3.connect(db_path) as conn:
        SQLquery = '''
        SELECT id AS crop_id, noise_review_label FROM crops
        WHERE  noise_review_label IS NOT NULL
        
        '''
        metas = pd.read_sql_query(SQLquery, conn)
    
    feature_columns = lr_config['feature_columns']

    reviewed_data = pd.merge(left=metas, right=features, on='crop_id', how='inner')
    if len(reviewed_data) < lr_config['minimum_reviewed_samples']:
        logger.warning(f"已审核数据样本量{len(reviewed_data)}小于设定的最小值{lr_config['minimum_reviewed_samples']}，可能无法训练出有效的模型")
        return constants.STAGE_INTERRUPT # type: ignore
    logger.info(f"共有{len(reviewed_data)}条已审核数据")




    noise_positive_label = lr_config['noise_positive_label']
    excluding_label = lr_config['excluding_label']
    clean_label = lr_config['clean_label']

    training_labels = set(noise_positive_label) | {clean_label}
    filtered = reviewed_data[reviewed_data['noise_review_label'].isin(training_labels)].copy()
    skipped_labels = sorted(set(reviewed_data['noise_review_label']) - training_labels - set(excluding_label))
    if skipped_labels:
        logger.warning("以下人工标签未纳入logistic regression训练: %s", skipped_labels)

    X = filtered[feature_columns].astype(float).copy()
    y = filtered['noise_review_label'].isin(noise_positive_label).to_numpy(dtype=np.int64)
    class_counts = np.bincount(y, minlength=2)
    if class_counts.min() < 2:
        logger.warning("正负样本数量不足，无法进行分层训练: clean=%d, noise=%d", class_counts[0], class_counts[1])
        return constants.STAGE_INTERRUPT  # type: ignore

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=lr_config['test_ratio'],
        random_state=42,
        stratify=y,
    )
    # holdout test集，以免在小样本上过拟合，无法评估模型在新数据上的表现
    k_fold = min(lr_config['fold_num'], int(np.bincount(y_train, minlength=2).min()))
    if k_fold < 2:
        logger.warning("训练集正负样本数量不足，无法进行交叉验证。")
        return constants.STAGE_INTERRUPT  # type: ignore
    logger.info(f"{k_fold}KFold训练集，测试集size分别为{len(X_train)}和{len(X_test)}")

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=lr_config['lr_C'],
            solver=lr_config['solver'],
            class_weight=lr_config['class_weight'],
            random_state=42,
            max_iter=lr_config['max_iter'],
        ),
    )


    cv = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    prob = cross_val_predict(model, X_train, y_train, cv=cv, method='predict_proba')[:, 1]

    
    # GridSearch对threshold进行调优，选取在clean recall达到指定最小值的前提下noise recall最高的threshold
    thresholds = np.linspace(0.05, 0.95, 19)
    clean_recall_min = lr_config['minimum_clean_recall']
    logger.info(f"开始网格搜索调优threshold，要求clean recall至少达到{clean_recall_min}")
    best_threshold = 0
    max_noise_recall = 0
    for threshold in thresholds:
        pred = (prob >= threshold).astype(int)
        confusion_matrix(y_train, pred)
        tn, fp, fn, tp = confusion_matrix(y_train, pred).ravel()
        clean_recall = tn / (tn + fp)
        if clean_recall >= clean_recall_min:
            noise_recall = tp / (tp + fn)
            if noise_recall > max_noise_recall:
                max_noise_recall = noise_recall
                best_threshold = threshold
    logger.info(f"调优后最佳threshold: {best_threshold:.2f}, Noise样本的recall: {max_noise_recall:.4f}")

    pred = (prob >= best_threshold).astype(int)
    logger.info("训练集性能评估:")
    logger.info(classification_report(y_train, pred,))
    logger.info(f"训练集上的ROC AUC: {roc_auc_score(y_train, prob)}")
    
    
    model = make_pipeline(
        StandardScaler(),
        LogisticRegressionWithThreshold(
            LogisticRegression(
                C=lr_config['lr_C'],
                solver=lr_config['solver'],
                class_weight=lr_config['class_weight'],
                random_state=42,
                max_iter=lr_config['max_iter'],
            ), 
            threshold=best_threshold),
    )
    # 在测试集上评估模型性能
    model.fit(X_train, y_train)
    test_pred = (model.predict(X_test))
    logger.info("测试集性能评估:")
    logger.info(classification_report(y_test, test_pred))
    logger.info(f"测试集上的ROC AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")

    model_name = lr_config['model_name_prefix'] + time.strftime("%Y%m%d-%H%M", time.localtime()) + ".joblib"
    joblib.dump(model, utils.join_data_root(config['path']['model_dir']) / model_name)
    with open(utils.join_data_root(config['path']['model_dir']) / lr_config['model_pointer_path'], "w") as f:
        f.write(model_name)
    logger.info(f"模型已保存: {model_name}，并更新最新指针文件: {lr_config['model_pointer_path']}")
    

if __name__ == "__main__":
    main()
