import httpx
#from bs4 import BeautifulSoup

import json
import pandas as pd
import numpy as np
import os

import time

import torch

from pathlib import Path
import sqlite3
from PIL import Image, ImageOps
from transformers import AutoModel
from accelerate import Accelerator
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from transformers import AutoImageProcessor
from transformers.image_utils import load_image
load_dotenv(override=False)
from huggingface_hub import login
import utils
import constants

logger = utils.get_logger("stage_10_train_loss_tracking")


# 路径缓存加载helper

def get_torch_device() -> torch.device:
    print(
        f" Metal Availablility: {torch.backends.mps.is_available()}\n"
        f"Cuda Availability: {torch.cuda.is_available()}\n"
        f"Device count: {torch.cuda.device_count()}"
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def resolve_feature_cache_file(file_name: str | None, latest_feature_cache_path: Path) -> str:
    if not file_name:
        raise ValueError("未配置 noise_detection.active_feature_cache_file")
    if file_name != "latest":
        return file_name
    if not latest_feature_cache_path.exists():
        raise FileNotFoundError(f"找不到最新特征缓存指针文件: {latest_feature_cache_path}")
    latest_file = latest_feature_cache_path.read_text(encoding="utf-8").strip()
    if not latest_file:
        raise ValueError(f"最新特征缓存指针文件为空: {latest_feature_cache_path}")
    return latest_file




# ================= 线性分类头定义 =================

from torch import nn
class LinearHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


def make_sawtooth_lambda(lr_max: float, lr_min: float, period_steps: int):
    """
    Return a LambdaLR-compatible function.

    At step 0: lr = lr_max
    At step period_steps - 1: approximately lr_min
    At step period_steps: jumps back to lr_max
    """
    if period_steps <= 1:
        raise ValueError("period_steps 必须大于 1")
    if lr_max <= 0 or lr_min < 0:
        raise ValueError("lr_max 必须大于 0，且 lr_min 必须大于等于 0")
    if lr_min > lr_max:
        raise ValueError("lr_min 必须小于等于 lr_max")

    min_ratio = lr_min / lr_max

    def lr_lambda(step: int):
        t = step % period_steps
        progress = t / (period_steps - 1)  # 0 -> 1
        ratio = 1.0 - progress * (1.0 - min_ratio)
        return ratio

    return lr_lambda



class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, feature_cache: dict):
            self.features = feature_cache["features"].float()
            self.labels = feature_cache["labels"].long()
            self.crop_ids = feature_cache["crop_ids"].long()

            assert len(self.features) == len(self.labels) == len(self.crop_ids)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx], self.crop_ids[idx]


def load_current_crop_labels(
    db_path: Path,
    crop_ids: list[int],
    noise_detection_cfg: dict,
    prediction_overlay: pd.DataFrame | None = None,
    chunk_size: int = 900,
) -> pd.DataFrame:
    # 训练标签采用 crop 级人工纠正优先；没有纠正时回退到当前配置的标签粒度。
    label_granularity = noise_detection_cfg["label_granularity"]
    if label_granularity == "submodel":
        base_label_expr = "COALESCE(i.submodel, i.fine_grained_series, c.series)"
    elif label_granularity == "fine_grained_series":
        base_label_expr = "COALESCE(i.fine_grained_series, c.series)"
    elif label_granularity == "series":
        base_label_expr = "c.series"
    else:
        raise ValueError(
            "noise_detection.label_granularity 必须是以下值之一: "
            "series, fine_grained_series, submodel"
        )
    label_expr = f"COALESCE(NULLIF(c.manual_corrected_label, ''), {base_label_expr})"

    rows = []
    with sqlite3.connect(db_path) as conn:
        for start in range(0, len(crop_ids), chunk_size):
            chunk = crop_ids[start:start + chunk_size]
            placeholders = ",".join(["?"] * len(chunk))
            sql = f"""
                SELECT
                    c.id AS crop_id,
                    {label_expr} AS label,
                    c.noise_review_label,
                    c.manual_corrected_label,
                    c.noise_predicted_label,
                    c.noise_predicted_prob
                FROM crops c
                JOIN images i ON i.id = c.image_id
                WHERE c.id IN ({placeholders})
            """
            rows.append(pd.read_sql_query(sql, conn, params=chunk))

    if not rows:
        return pd.DataFrame(columns=["crop_id", "label", "filter_reason"])
    labels = pd.concat(rows, ignore_index=True)
    labels["crop_id"] = labels["crop_id"].astype(int)
    if prediction_overlay is not None and not prediction_overlay.empty:
        # sync_to_db=false 时，用上一轮预测 CSV 覆盖 DB 中可能为空的预测字段。
        labels = labels.drop(columns=["noise_predicted_label", "noise_predicted_prob"])
        labels = labels.merge(prediction_overlay, on="crop_id", how="left")
    labels["filter_reason"] = ""

    corrected = (
        labels["manual_corrected_label"].notna()
        & (labels["manual_corrected_label"].astype(str).str.strip() != "")
    )
    review_label = labels["noise_review_label"].fillna("").astype(str).str.strip()

    if noise_detection_cfg["exclude_manual_noise"]:
        # 人工确认错标但已给出正确标签的样本保留，并用纠正标签训练。
        manual_noise_labels = set(noise_detection_cfg["manual_noise_labels"])
        manual_noise = review_label.isin(manual_noise_labels)
        corrected_wrong_label = (
            review_label.eq(constants.NOISE_REVIEW_LABEL_WRONG_LABEL)
            & corrected
        )
        manual_excluded = manual_noise & ~corrected_wrong_label
        labels.loc[manual_excluded, "filter_reason"] = "manual_noise"

    if noise_detection_cfg["exclude_predicted_noise"]:
        # 上一轮模型预测噪声只过滤未人工确认 OK、也未人工纠正的样本。
        predicted_noise_labels = set(noise_detection_cfg["predicted_noise_labels"])
        predicted_prob = pd.to_numeric(labels["noise_predicted_prob"], errors="coerce").fillna(0.0)
        predicted_excluded = (
            labels["filter_reason"].eq("")
            & ~corrected
            & review_label.ne(constants.NOISE_REVIEW_LABEL_OK)
            & labels["noise_predicted_label"].isin(predicted_noise_labels)
            & (predicted_prob >= float(noise_detection_cfg["predicted_noise_min_prob"]))
        )
        labels.loc[predicted_excluded, "filter_reason"] = "predicted_noise"

    return labels


def attach_current_labels_to_feature_cache(
    feature_cache: dict,
    db_path: Path,
    noise_detection_cfg: dict,
    prediction_overlay: pd.DataFrame | None = None,
) -> dict:
    features = feature_cache["features"].float()
    crop_ids = feature_cache["crop_ids"].long()
    crop_id_list = [int(crop_id) for crop_id in crop_ids.tolist()]
    labels_df = load_current_crop_labels(
        db_path=db_path,
        crop_ids=crop_id_list,
        noise_detection_cfg=noise_detection_cfg,
        prediction_overlay=prediction_overlay,
    )
    order_df = pd.DataFrame({"crop_id": crop_id_list, "feature_index": range(len(crop_id_list))})
    labeled = order_df.merge(labels_df, on="crop_id", how="left")
    labeled["label"] = labeled["label"].astype("string").str.strip()
    labeled["filter_reason"] = labeled["filter_reason"].fillna("")
    keep = labeled["label"].notna() & (labeled["label"] != "") & (labeled["filter_reason"] == "")
    if not keep.any():
        raise ValueError("特征缓存中没有可用于本轮训练的已标注 crop")

    skipped_unlabeled = int((labeled["label"].isna() | (labeled["label"] == "")).sum())
    skipped_filtered = int((labeled["filter_reason"] != "").sum())
    labeled = labeled.loc[keep].copy()
    feature_indices = torch.tensor(labeled["feature_index"].to_numpy(), dtype=torch.long)
    labels = labeled["label"].astype(str)
    label_names = sorted(labels.unique())
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    return {
        **feature_cache,
        "features": features.index_select(0, feature_indices),
        "crop_ids": torch.tensor(labeled["crop_id"].to_numpy(), dtype=torch.long),
        "labels": torch.tensor(labels.map(label_to_id).to_numpy(), dtype=torch.long),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "skipped_unlabeled_count": skipped_unlabeled,
        "skipped_filtered_count": skipped_filtered,
    }


def load_prediction_overlay(config: dict) -> pd.DataFrame | None:
    noise_detection_cfg = config["noise_detection"]
    if not noise_detection_cfg["exclude_predicted_noise"]:
        return None
    if config["lr_prediction"]["sync_to_db"]:
        return None

    prediction_dir = utils.get_current_loss_round_dir(config)
    prediction_path = prediction_dir / config["lr_prediction"]["prediction_file_name"]
    if not prediction_path.exists():
        logger.warning(
            "lr_prediction.sync_to_db=false，但上一轮预测文件不存在，跳过预测噪声过滤: %s",
            prediction_path,
        )
        return None

    predictions = pd.read_csv(prediction_path)
    required_columns = {"crop_id", "noise_predicted_label", "noise_predicted_prob"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"LR 预测 CSV 缺少必要列: {sorted(missing_columns)}")
    predictions = predictions[["crop_id", "noise_predicted_label", "noise_predicted_prob"]].copy()
    predictions["crop_id"] = predictions["crop_id"].astype(int)
    predictions = predictions.drop_duplicates("crop_id", keep="last")
    logger.info("从上一轮预测 CSV 加载 %d 条噪声预测，用于本轮训练过滤: %s", len(predictions), prediction_path)
    return predictions


def save_label_map(
    loss_dir: Path,
    label_granularity: str,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
) -> Path:
    label_map_path = loss_dir / "label_map.json"
    payload = {
        "label_granularity": label_granularity,
        "label_to_id": label_to_id,
        "id_to_label": {str(idx): label for idx, label in id_to_label.items()},
    }
    label_map_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return label_map_path




def main(config: dict | None = None) -> None:
    if config is None:
        config = utils.load_pipeline_config()

    utils.init_db(config=config)
    noise_detection_cfg = config["noise_detection"]
    loss_tracking_cfg = config["loss_noise_tracking"]
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    device = get_torch_device()
    feature_cache_dir = utils.join_data_root(
        noise_detection_cfg.get("feature_cache_dir", "feature_cache"),
        config=config,
    )
    active_feature_cache_file = noise_detection_cfg["active_feature_cache_file"]
    latest_feature_cache_path = feature_cache_dir / noise_detection_cfg.get(
        "latest_feature_cache_file",
        "latest_feature_cache.txt",
    )
    
    # 先创建当前轮次目录，但等本轮训练产物完整写入后再更新 latest 指针。
    loss_dir = utils.create_new_loss_round_dir(config)
    loss_history_path = loss_dir / loss_tracking_cfg["loss_history_file_name"]
    epoch_history_path = loss_dir / loss_tracking_cfg["epoch_history_file_name"]
    
    model_dir = utils.join_data_root(config["path"].get("model_dir", "model"), config=config)
    model_checkpoint_prefix = loss_tracking_cfg.get("model_checkpoint_prefix", "DINO_CLS_HEAD")
    embed_dim = int(loss_tracking_cfg['embedding_feature_dim'])
    lr_max = float(loss_tracking_cfg['learning_rate_high'])
    lr_min = float(loss_tracking_cfg['learning_rate_low'])
    period = int(loss_tracking_cfg['lr_cycle_period'])
    epochs = int(loss_tracking_cfg['num_epochs'])
    weight_decay = float(loss_tracking_cfg['weight_decay'])
    
    logger.info("第 10 阶段启动，数据库路径: %s", db_path)
    

    feature_cache_file = resolve_feature_cache_file(active_feature_cache_file, latest_feature_cache_path)
    raw_feature_cache = torch.load(feature_cache_dir / feature_cache_file, map_location="cpu")
    prediction_overlay = load_prediction_overlay(config)
    feature_cache = attach_current_labels_to_feature_cache(
        raw_feature_cache,
        db_path=db_path,
        noise_detection_cfg=noise_detection_cfg,
        prediction_overlay=prediction_overlay,
    )
    label_to_id = feature_cache['label_to_id']
    id_to_label = feature_cache['id_to_label']
    label_map_path = save_label_map(
        loss_dir,
        noise_detection_cfg["label_granularity"],
        label_to_id,
        id_to_label,
    )
    logger.info(
        "已加载特征缓存 %s，当前可训练样本=%d，跳过无标签样本=%d，按噪声过滤跳过=%d，标签类别数=%d，label map=%s",
        feature_cache_file,
        len(feature_cache["features"]),
        feature_cache["skipped_unlabeled_count"],
        feature_cache["skipped_filtered_count"],
        len(label_to_id),
        label_map_path,
    )

    

    feature_dataset = FeatureDataset(feature_cache)
    head_batch_size = int(noise_detection_cfg.get("linear_head_train_batch_size", 32))
    feature_dataloader = torch.utils.data.DataLoader(feature_dataset, 
                                                batch_size=head_batch_size, shuffle=True)

    
    
    train_device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    head = LinearHead(input_dim=embed_dim, num_classes=len(label_to_id)).to(train_device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr_max, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_sawtooth_lambda(lr_max, lr_min, period),
    )
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    logger.info("已建立 %d 维线性分类头，训练设备: %s，优化器: AdamW，学习率周期: %d 步", embed_dim, train_device, period)
    
    
    # === 主训练部分 ===
    # 定义数据loss_record = (epoch, crop_id, label_id, pred_id, pred_confidence, correct, loss_value)
    loss_records = []
    epoch_records = []
    global_step = 0
    for epoch in tqdm(range(epochs), desc="训练线性头", unit="epoch"):
        head.train()
        
        running_loss = 0.0
        running_correct = 0
        running_n = 0
        
        
        for x_cpu, y_cpu, crop_ids_cpu in feature_dataloader:
            x_dev = x_cpu.to(train_device)
            y_dev = y_cpu.to(train_device)
            
            optimizer.zero_grad(set_to_none=True)
            logits_dev = head(x_dev)
            loss_values_dev = criterion(logits_dev, y_dev)
            loss_dev = loss_values_dev.mean()
            loss_dev.backward()
            optimizer.step()
            scheduler.step()
            
            
            # 按batch记录全程loss，预测情况，正确性，softmax后的置信度
            with torch.no_grad():
                probs_dev = torch.softmax(logits_dev, dim=1)
                pred_confidence_dev, predict_dev = probs_dev.max(dim=1)
                correct_dev = predict_dev.eq(y_dev)
            
            loss_values_cpu = loss_values_dev.detach().cpu()
            predict_cpu = predict_dev.detach().cpu()
            pred_confidence_cpu = pred_confidence_dev.detach().cpu()
            correct_cpu = correct_dev.detach().cpu()
            y_cpu = y_dev.detach().cpu()
            crop_ids_cpu = crop_ids_cpu.cpu()
            
            #写入tuple记录
            for i, cropid in enumerate(crop_ids_cpu.tolist()):
                loss_records.append(
                    (   epoch, 
                        cropid, 
                        y_cpu[i].item(), 
                        predict_cpu[i].item(), 
                        pred_confidence_cpu[i].item(), 
                        correct_cpu[i].item(), 
                        loss_values_cpu[i].item())
                )

            running_loss += float(loss_values_cpu.sum())
            running_correct += int(correct_cpu.sum())
            running_n += int(y_cpu.numel())
            global_step += 1
        # epoch信息记录
        epoch_records.append({
            "epoch": epoch,
            "loss": running_loss / max(1, running_n),
            "accuracy": running_correct / max(1, running_n),
            "n": running_n,
        })
        
    loss_history = pd.DataFrame(
        loss_records,
        columns=["epoch", "crop_id", "label_id", "pred_id", "pred_confidence", "correct", "loss_value"],
    )
    epoch_history = pd.DataFrame(epoch_records)
    
    loss_history.to_csv(loss_history_path, index=False)
    epoch_history.to_csv(epoch_history_path, index=False)

    timestamp = time.strftime("%Y%m%d-%H%M", time.localtime())
    model_checkpoint_name = f"{model_checkpoint_prefix}_{len(label_to_id)}classes_{timestamp}.pt"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_checkpoint_path = model_dir / model_checkpoint_name
    torch.save(head.state_dict(), model_checkpoint_path)

    logger.info("loss history 已保存至 %s", loss_history_path)
    logger.info("epoch history 已保存至 %s", epoch_history_path)
    logger.info("线性分类头 checkpoint 已保存至 %s", model_checkpoint_path)
    utils.update_latest_loss_round_pointer(config, loss_dir)
    logger.info("最新 loss 轮次指针已更新为 %s", loss_dir.name)

if __name__ == "__main__":
    main()
