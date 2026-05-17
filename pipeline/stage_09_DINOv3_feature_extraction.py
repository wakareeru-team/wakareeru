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

logger = utils.get_logger("stage_09_DINOv3_feature_extraction")


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



# 图片在线裁剪Helper
def load_crop_manifest(
    db_path,
    series: list[str] | None = None,
    power_type: str | None = None,
    crop_status: str | None = None,
    min_score: float | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    label_granularity='fine_grained_series',
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



def _source_image_path(
    row: pd.Series | dict,
    img_root: Path | None = None,
    config: dict | None = None,
) -> Path:
    path = Path(str(row["downloaded_path"]).replace("\\", "/"))
    if path.is_absolute():
        return path
    if img_root is not None:
        return img_root / path
    return utils.join_data_root(path, config=config)


def _expanded_box(row: pd.Series | dict, image_size: tuple[int, int], pad_frac: float = 0.04):
    width, height = image_size
    x1, y1, x2, y2 = (float(row[k]) for k in ["box_x1", "box_y1", "box_x2", "box_y2"])
    pad = max(x2 - x1, y2 - y1) * pad_frac
    left = max(0, int(np.floor(x1 - pad)))
    top = max(0, int(np.floor(y1 - pad)))
    right = min(width, int(np.ceil(x2 + pad)))
    bottom = min(height, int(np.ceil(y2 + pad)))
    if right <= left or bottom <= top:
        raise ValueError(f"bad crop box for crop_id={row.get('crop_id', row.get('id'))}: {(left, top, right, bottom)}")
    return left, top, right, bottom


def crop_from_image(
    img: Image.Image,
    row: pd.Series | dict,
    pad_frac: float = 0.04,
) -> Image.Image:
    return img.crop(_expanded_box(row, img.size, pad_frac=pad_frac))


def load_crop(
    row: pd.Series | dict,
    img_root: Path | None = None,
    config: dict | None = None,
    pad_frac: float = 0.04,
) -> Image.Image:
    img = utils.load_img_with_orientation(
        _source_image_path(row, img_root=img_root, config=config)
    )
    return crop_from_image(img, row, pad_frac=pad_frac)


# ===================== huggingface 数据集 ===================
class CropImageDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            df_crops: pd.DataFrame,
            img_root,
            pad_frac: float = 0.04):
        self.df_crops = df_crops.reset_index(drop=True)
        self.img_root = img_root
        self.pad_frac = pad_frac
    
    def __len__(self):
        return len(self.df_crops)
    
    
    def __getitem__(self, idx):
        row = self.df_crops.iloc[idx]
        cropped_img = load_crop(row, img_root=self.img_root, pad_frac=self.pad_frac)
        return cropped_img, row.to_dict()


class FeatureCollator:
    def __init__(self, processor, label_to_id: dict[str, int]):
        self.processor = processor
        self.label_to_id = label_to_id

    def __call__(self, batch):
        images, metas = zip(*batch)
        image_tensors = self.processor(images=list(images), return_tensors="pt")["pixel_values"]
        labels = torch.tensor([self.label_to_id[str(meta["label"])] for meta in metas], dtype=torch.long)
        crop_ids = torch.tensor([int(meta["crop_id"]) for meta in metas], dtype=torch.long)
        return image_tensors, labels, crop_ids









def main(config=None):
    
    if config is None:
        config = utils.load_pipeline_config()
    utils.init_db(config=config)
    login(token=os.getenv("HF_TOKEN"))
    noise_detection_cfg = config['noise_detection']
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    device = get_torch_device()
    label_granularity = noise_detection_cfg.get("label_granularity", "fine_grained_series")
    feature_cache_dir = utils.join_data_root(
        noise_detection_cfg.get("feature_cache_dir", "feature_cache"),
        config=config,
    )
    active_feature_cache_file = noise_detection_cfg.get("active_feature_cache_file", "latest")
    latest_feature_cache_path = feature_cache_dir / noise_detection_cfg.get(
        "latest_feature_cache_file",
        "latest_feature_cache.txt",
    )
    loss_history_path = utils.join_data_root(
        noise_detection_cfg.get("loss_history_path", "demo_loss_history.csv"),
        config=config,
    )
    epoch_history_path = utils.join_data_root(
        noise_detection_cfg.get("epoch_history_path", "demo_epoch_history.csv"),
        config=config,
    )
    logger.info("Starting DINOv3 feature extraction...")
    
    
    # 加载数据
    df_crops = load_crop_manifest(
    db_path=db_path,
    series= None if noise_detection_cfg.get("full_series") else noise_detection_cfg.get("series_test_scope"),
    label_granularity=label_granularity,
    )
    
    pretrained_model_name = noise_detection_cfg["hf_model_name"]
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name, 
        device_map="auto", 
    )
    logger.info(f'已加载特征提取模型: {pretrained_model_name}，在加速设备上运行: {device}。准备开始特征提取...')
    
    
    labels = sorted(df_crops['label'].dropna().astype(str).unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
        
    
    embed_dim = model.config.hidden_size
    logger.info(f"模型输出特征维度: {embed_dim}。开始处理 {len(df_crops)} 个crop...")
    
    
    
    batch_size = noise_detection_cfg.get("feature_extraction_batch_size", 16)
    dataset = CropImageDataset(
        df_crops,
        img_root=utils.get_data_root(config),
        pad_frac=noise_detection_cfg["crop_pad_frac"],
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=FeatureCollator(processor, label_to_id),
        num_workers=noise_detection_cfg.get("extration_workers", 0),
    )
    
    
    all_feature = []
    all_labels = []
    all_crop_ids = []

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            
            image_tensors, labels, crop_ids = batch
            image_tensors = image_tensors.to(model.device)
            outputs = model(image_tensors)
            
            pooled_output = outputs.pooler_output #CLS output
            
            all_feature.append(pooled_output.cpu())
            all_labels.append(labels)
            all_crop_ids.append(crop_ids)

    feature_cache = {
        "features": torch.cat(all_feature, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "crop_ids": torch.cat(all_crop_ids, dim=0),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "model_name": pretrained_model_name,
    }


    # saving

    data_length = df_crops.shape[0]
    class_num = len(label_to_id)
    logger.info(f"特征提取完成。共处理 {len(df_crops)} 个crop，提取标签共{class_num}。开始保存特征缓存...")
    file_name = f"demo_dinov3_features_{data_length}crops_{class_num}labels.pt"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(feature_cache, feature_cache_dir / file_name)
    latest_feature_cache_path.write_text(file_name, encoding="utf-8")
    logger.info(f"特征缓存保存完成: {feature_cache_dir / file_name}，并更新latest指针。")

if __name__ == "__main__":
    main()
