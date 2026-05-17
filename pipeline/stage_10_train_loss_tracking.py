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
        raise ValueError("noise_detection.active_feature_cache_file is not configured")
    if file_name != "latest":
        return file_name
    if not latest_feature_cache_path.exists():
        raise FileNotFoundError(f"Latest feature cache pointer not found: {latest_feature_cache_path}")
    latest_file = latest_feature_cache_path.read_text(encoding="utf-8").strip()
    if not latest_file:
        raise ValueError(f"Latest feature cache pointer is empty: {latest_feature_cache_path}")
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
        raise ValueError("period_steps must be > 1")
    if lr_max <= 0 or lr_min < 0:
        raise ValueError("lr_max must be > 0 and lr_min must be >= 0")
    if lr_min > lr_max:
        raise ValueError("lr_min must be <= lr_max")

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
    loss_history_path = utils.join_data_root(
        loss_tracking_cfg.get("loss_history_path", "demo_loss_history.csv"),
        config=config,
    )
    epoch_history_path = utils.join_data_root(
        loss_tracking_cfg.get("epoch_history_path", "demo_epoch_history.csv"),
        config=config,
    )
    model_dir = utils.join_data_root(loss_tracking_cfg.get("model_dir", "model"), config=config)
    model_checkpoint_prefix = loss_tracking_cfg.get("model_checkpoint_prefix", "DINO_CLS_HEAD")
    embed_dim = int(loss_tracking_cfg['embedding_feature_dim'])
    lr_max = float(loss_tracking_cfg['learning_rate_high'])
    lr_min = float(loss_tracking_cfg['learning_rate_low'])
    period = int(loss_tracking_cfg['lr_cycle_period'])
    epochs = int(loss_tracking_cfg['num_epochs'])
    weight_decay = float(loss_tracking_cfg['weight_decay'])
    
    logger.info("Stage 10 started. DB path: %s", db_path)
    

    feature_cache_file = resolve_feature_cache_file(active_feature_cache_file, latest_feature_cache_path)
    feature_cache = torch.load(feature_cache_dir / feature_cache_file, map_location="cpu")
    label_to_id = feature_cache['label_to_id']
    id_to_label = feature_cache['id_to_label']
    logger.info(f'已加载特征缓存，包含 {len(feature_cache["features"])} 个样本，标签类别数: {len(label_to_id)}')

    

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
    logger.info(f'已建立{embed_dim}维线性分类头，训练设备: {train_device}，优化器: AdamW，学习率周期: {period} steps')
    
    
    # === 主训练部分 ===
    # 定义数据loss_record = (epoch, crop_id, label_id, pred_id, pred_confidence, correct, loss_value)
    loss_records = []
    epoch_records = []
    global_step = 0
    for epoch in tqdm(range(epochs), desc="Training epochs"):
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
    
    loss_history_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_history_path.parent.mkdir(parents=True, exist_ok=True)
    loss_history.to_csv(loss_history_path, index=False)
    epoch_history.to_csv(epoch_history_path, index=False)

    timestamp = time.strftime("%Y%m%d-%H%M", time.localtime())
    model_checkpoint_name = f"{model_checkpoint_prefix}_{len(label_to_id)}classes_{timestamp}.pt"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_checkpoint_path = model_dir / model_checkpoint_name
    torch.save(head.state_dict(), model_checkpoint_path)

    logger.info("Loss history saved to %s", loss_history_path)
    logger.info("Epoch history saved to %s", epoch_history_path)
    logger.info("Linear head checkpoint saved to %s", model_checkpoint_path)

if __name__ == "__main__":
    main()
