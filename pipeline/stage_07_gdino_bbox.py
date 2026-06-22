import gc
import os
import sqlite3

import pandas as pd
import torch
from accelerate import Accelerator
from dotenv import load_dotenv
from huggingface_hub import login
from torchvision.ops import batched_nms
from tqdm.auto import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.utils.generic import ModelOutput

import utils

config = utils.load_pipeline_config()

logger = utils.get_logger("stage_07_gdino_bbox")
IMAGE_DB_PATH = utils.join_data_root(config["path"]["db_path"], config=config)


def detach_to_cpu(value):
    """递归地把模型输出里的 tensor 搬到 CPU，避免缓存结果时继续占用显存。"""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, ModelOutput):
        return value.__class__(**{key: detach_to_cpu(item) for key, item in value.items()})
    if isinstance(value, dict):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [detach_to_cpu(item) for item in value]
    return value


def empty_accelerator_cache():
    """释放当前加速后端未占用的缓存内存。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def gdino_text_label_for_power_type(power_type):
    # Grounding-DINO processor 需要每张图对应一个 label 列表，所以这里返回 ["..."] 而不是普通字符串。
    if power_type in ["Steam Locomotive", "Diesel Locomotive", "Electric Locomotive"]:
        return ["a single locomotive"]
    return ["a train"]


def gdino_detection_with_preloaded_images(processor, model, device, PIL_images, labels, pbar=None):
    """对预载的 PIL 图片列表运行 GDINO 推理。

    返回 (outputs_cpu, token_ids_cpu, target_sizes)。
    token_ids 是 processor 的 tokenizer 输出，后处理时需要用到。
    target_sizes 由图片尺寸推导，格式为 [(h, w), ...]。
    """
    target_sizes = [img.size[::-1] for img in PIL_images]
    with torch.inference_mode():
        inputs = processor(
            images=PIL_images,
            text=labels,
            return_tensors="pt",
            padding=True,
        ).to(device)
        outputs = detach_to_cpu(model(**inputs))
    token_ids = inputs["input_ids"].detach().cpu()
    del inputs
    empty_accelerator_cache()
    if pbar is not None:
        pbar.update(len(PIL_images))
    return outputs, token_ids, target_sizes


def threshold_gdino_outputs(processor, outputs, token_ids, target_sizes, box_threshold=0.2, text_threshold=0.2):
    """对 gdino_detection_with_preloaded_images 的输出做阈值后处理，返回每张图的检测结果列表。"""
    return processor.post_process_grounded_object_detection(
        outputs,
        token_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_sizes,
    )


def nms_postprocess(results, iou_threshold=0.5):
    """对 threshold_gdino_outputs 的结果做 label-aware NMS，返回与输入等长的结果列表。

    每个结果 dict 比输入多一个 keep_indices 字段，记录 NMS 前的原始索引。
    """
    nms_results = []
    for result in results:
        boxes = result["boxes"]
        scores = result["scores"]
        text_labels = result["text_labels"]

        if len(boxes) == 0:
            nms_results.append({"boxes": boxes, "scores": scores, "text_labels": text_labels, "keep_indices": []})
            continue

        label_to_id = {label: idx for idx, label in enumerate(dict.fromkeys(text_labels))}
        label_ids = torch.tensor([label_to_id[label] for label in text_labels], dtype=torch.long)
        keep = batched_nms(boxes.float(), scores, label_ids, iou_threshold)

        nms_results.append({
            "boxes": boxes[keep],
            "scores": scores[keep],
            "text_labels": [text_labels[i] for i in keep.tolist()],
            "keep_indices": keep.tolist(),
        })

    return nms_results


_UPSERT_CROPS_SQL = """
INSERT INTO crops (
    image_id, series, power_type, crop_index, source_result_index,
    detector_model, detector_label, detector_score,
    box_x1, box_y1, box_x2, box_y2, box_area, nms_iou_threshold
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(image_id, detector_model, nms_iou_threshold, crop_index) DO UPDATE SET
    series              = excluded.series,
    power_type          = excluded.power_type,
    source_result_index = excluded.source_result_index,
    detector_label      = excluded.detector_label,
    detector_score      = excluded.detector_score,
    box_x1              = excluded.box_x1,
    box_y1              = excluded.box_y1,
    box_x2              = excluded.box_x2,
    box_y2              = excluded.box_y2,
    box_area            = excluded.box_area,
    crop_status         = 'pending',
    crop_reason         = NULL,
    updated_at          = CURRENT_TIMESTAMP
"""


def upsert_crops(conn, batch_df, nms_results, model_id, nms_iou_threshold):
    """把 nms_postprocess 的结果写入 crops 表（upsert）。

    batch_df 与 nms_results 必须等长且顺序对齐。
    """
    rows = []
    for (_, image_row), result in zip(batch_df.iterrows(), nms_results):
        image_id = int(image_row["id"])
        series = None if pd.isna(image_row["series"]) else str(image_row["series"])
        power_type = None if pd.isna(image_row["power_type"]) else str(image_row["power_type"])

        for crop_index, (source_idx, box, score, label) in enumerate(
            zip(result["keep_indices"], result["boxes"], result["scores"], result["text_labels"])
        ):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            rows.append((
                image_id, series, power_type,
                crop_index, int(source_idx),
                model_id, str(label), float(score),
                x1, y1, x2, y2,
                max(0.0, x2 - x1) * max(0.0, y2 - y1),
                float(nms_iou_threshold),
            ))

    conn.executemany(_UPSERT_CROPS_SQL, rows)
    conn.commit()


def main(config: dict | None = None):
    load_dotenv(override=True)
    if not config:
        config = utils.load_pipeline_config()
    login(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))
    utils.init_db(config=config)

    gdino_cfg = config["gdino"]
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    model_id = gdino_cfg["model_name"]
    box_threshold = gdino_cfg["box_threshold"]
    text_threshold = gdino_cfg["text_threshold"]
    nms_iou_threshold = gdino_cfg["nms_iou_threshold"]
    inner_batch_size = gdino_cfg["batch_size"]
    outer_batch_size = gdino_cfg.get("outer_batch_size", 160)
    reprocess = gdino_cfg.get("reprocess", False)

    logger.info("使用基座模型：%s", model_id)
    logger.info(
        "MPS available: %s; CUDA available: %s; CUDA devices: %d",
        torch.backends.mps.is_available(),
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )

    device = Accelerator().device
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()

    with sqlite3.connect(db_path) as conn:
        full_df = pd.read_sql_query(
            """
            SELECT id, downloaded_path, series, power_type
            FROM images
            WHERE excluded = 0 AND download_status = 'downloaded'
              AND (cropped = 0 OR ?)
            """,
            conn,
            params=[int(reprocess)],
        )

    full_df["gdino_labels"] = full_df["power_type"].apply(gdino_text_label_for_power_type)
    full_df["abs_path"] = full_df["downloaded_path"].apply(
        lambda x: str(utils.join_data_root(str(x).replace("\\", "/"), config=config))
    )
    logger.info("待处理图片数：%d", len(full_df))

    with sqlite3.connect(db_path) as conn:
        with tqdm(total=len(full_df), desc="GDINO 推理", unit="img") as pbar:
            for outer_start in range(0, len(full_df), outer_batch_size):
                outer_df = full_df.iloc[outer_start : outer_start + outer_batch_size]
                outer_images = [utils.load_img_with_orientation(p) for p in outer_df["abs_path"]]
                outer_labels = outer_df["gdino_labels"].tolist()

                outer_results = []
                for inner_start in range(0, len(outer_df), inner_batch_size):
                    inner_images = outer_images[inner_start : inner_start + inner_batch_size]
                    inner_labels = outer_labels[inner_start : inner_start + inner_batch_size]

                    outputs, token_ids, target_sizes = gdino_detection_with_preloaded_images(
                        processor, model, device, inner_images, inner_labels, pbar=pbar
                    )
                    batch_results = threshold_gdino_outputs(
                        processor, outputs, token_ids, target_sizes, box_threshold, text_threshold
                    )
                    outer_results.extend(nms_postprocess(batch_results, nms_iou_threshold))
                    del outputs, token_ids, target_sizes, batch_results, inner_images, inner_labels
                    empty_accelerator_cache()
                    gc.collect()

                del outer_images
                upsert_crops(conn, outer_df, outer_results, model_id, nms_iou_threshold)
                image_ids = outer_df["id"].tolist()
                conn.execute(
                    f"UPDATE images SET cropped = 1 WHERE id IN ({','.join('?' * len(image_ids))})",
                    image_ids,
                )
                conn.commit()
                del outer_results, outer_labels, outer_df, image_ids
                empty_accelerator_cache()
                gc.collect()

    logger.info("GDINO bbox 检测完成。")


if __name__ == "__main__":
    main()
