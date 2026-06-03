import json
import os
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

import constants
import utils

import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from lr_model import register_legacy_main_alias

logger = utils.get_logger("stage_14_store_crops")





def label_to_ascii(label: str, fallback: str = "label") -> str:
    """Convert a Japanese train label into a deterministic ASCII slug.

    This is intentionally rule-based rather than LLM-based so saved dataset
    paths remain stable across runs.
    """
    text = unicodedata.normalize("NFKC", str(label)).strip().lower()
    for src, dst in constants.LABEL_ASCII_REPLACEMENTS:
        text = text.replace(src.lower(), dst)
    text = re.sub(r"[^0-9a-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    text = re.sub(r"\b(kiha|kumoha|kuha|moha|saha|deha|roha|saro|moro|kani)_(\d)", r"\1\2", text)
    return text or fallback


def save_crop_image(
    *,
    source_image_path: str | Path,
    output_path: str | Path,
    box_x1: float,
    box_y1: float,
    box_x2: float,
    box_y2: float,
    pad_frac: float = 0.04,
    image_format: str | None = None,
    jpeg_quality: int = 95,
) -> Path:
    """Crop one explicit bbox from an image and save it to disk."""
    source_image_path = Path(source_image_path)
    output_path = Path(output_path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".jpeg")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = utils.load_img_with_orientation(source_image_path)
    box_width = float(box_x2) - float(box_x1)
    box_height = float(box_y2) - float(box_y1)
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid crop box: ({box_x1}, {box_y1}, {box_x2}, {box_y2})")

    pad = max(box_width, box_height) * float(pad_frac)
    left = max(0, int(float(box_x1) - pad))
    top = max(0, int(float(box_y1) - pad))
    right = min(img.width, int(float(box_x2) + pad))
    bottom = min(img.height, int(float(box_y2) + pad))
    if right <= left or bottom <= top:
        raise ValueError(
            "Padded crop box is outside image bounds: "
            f"({left}, {top}, {right}, {bottom}) for {source_image_path}"
        )

    crop = img.crop((left, top, right, bottom))
    if crop.mode != "RGB":
        crop = crop.convert("RGB")

    suffix = output_path.suffix.lower()
    save_format = image_format
    if save_format is None:
        save_format = "JPEG" if suffix in {".jpg", ".jpeg"} else suffix.lstrip(".").upper()

    save_kwargs = {}
    if save_format.upper() in {"JPEG", "JPG"}:
        save_format = "JPEG"
        save_kwargs["quality"] = int(jpeg_quality)
    crop.save(output_path, format=save_format, **save_kwargs)
    return output_path


def validate_config_column_names(columns: list[str]) -> None:
    unsafe = [col for col in columns if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", col)]
    if unsafe:
        raise ValueError(f"Unsafe configured column names: {unsafe}")


def build_en_label(
    metadata: pd.DataFrame,
    *,
    label_column: str,
) -> pd.DataFrame:
    if label_column not in metadata.columns:
        raise ValueError(f"image metadata中不存在label列: {label_column!r}")

    metadata = metadata.copy()
    metadata["label"] = metadata[label_column]
    metadata["label_en"] = metadata["label"].map(lambda label: label_to_ascii(label, fallback="unknown"))
    return metadata


def invalidate_metadata_for_manual_corrections(
    metadata: pd.DataFrame,
    *,
    corrected_mask: pd.Series,
    columns: list[str],
    label_column: str,
) -> pd.DataFrame:
    validate_config_column_names(columns)
    if label_column in columns:
        raise ValueError(
            "crops_storage.manual_correction_invalidate_metadata_columns "
            f"不能包含当前label列: {label_column!r}"
        )
    missing_columns = [col for col in columns if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"人工纠正后要清空的metadata列不存在: {missing_columns}")

    metadata = metadata.copy()
    if columns and corrected_mask.any():
        metadata.loc[corrected_mask, columns] = pd.NA
    return metadata


def build_label_tables(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = (
        metadata[["label", "label_en"]]
        .dropna(subset=["label"])
        .drop_duplicates()
        .sort_values(["label_en", "label"], kind="stable")
        .reset_index(drop=True)
    )
    labels.insert(0, "label_id", range(len(labels)))

    counts = metadata["label"].value_counts(dropna=False).rename("count").reset_index()
    counts.columns = ["label", "count"]
    labels = labels.merge(counts, on="label", how="left")

    metadata = metadata.merge(labels[["label", "label_id"]], on="label", how="left")
    metadata["label_id"] = metadata["label_id"].astype("Int64")
    return metadata, labels


def flush_crop_save_updates(db_path: Path, updates: list[tuple[str, int]]) -> None:
    if not updates:
        return
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "UPDATE crops SET saved = 1, crop_path = ? WHERE id = ?",
            updates,
        )
        conn.commit()


def write_dataset_manifest(
    *,
    manifest_path: Path,
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    crops_storage_config: dict,
) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "metadata_file": crops_storage_config["metadata_file_name"],
        "labels_file": crops_storage_config["labels_file_name"],
        "num_samples": int(len(metadata)),
        "num_labels": int(len(labels)),
        "label_column": crops_storage_config["label_column"],
        "image_extension": crops_storage_config["image_extension"],
        "crop_pad_frac": crops_storage_config["crop_pad_frac"],
        "manual_reviewed_count": int(metadata["manual_reviewed"].sum()),
        "manual_corrected_count": int(
            metadata["manual_corrected_label"].notna().sum()
            if "manual_corrected_label" in metadata.columns
            else 0
        ),
        "noise_filtering": {
            "exclude_manual_noise": crops_storage_config["exclude_manual_noise"],
            "manual_noise_labels": crops_storage_config["manual_noise_labels"],
            "exclude_predicted_noise": crops_storage_config["exclude_predicted_noise"],
            "noise_prediction_round": crops_storage_config["noise_prediction_round"],
            "noise_prediction_file_name": crops_storage_config["noise_prediction_file_name"],
            "predicted_noise_labels": crops_storage_config["predicted_noise_labels"],
            "predicted_noise_min_prob": crops_storage_config["predicted_noise_min_prob"],
            "manual_correction_invalidate_metadata_columns": crops_storage_config[
                "manual_correction_invalidate_metadata_columns"
            ],
        },
        "notes": "Generated by stage_14_store_crops.py",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_noise_predictions(config: dict, crops_storage_config: dict) -> pd.DataFrame:
    prediction_round = crops_storage_config["noise_prediction_round"]
    prediction_dir = utils.get_loss_round_dir(
        config=config,
        active_round=prediction_round,
    )
    prediction_path = prediction_dir / crops_storage_config["noise_prediction_file_name"]
    if not prediction_path.exists():
        raise FileNotFoundError(f"Expected LR prediction CSV not found: {prediction_path}")
    predictions = pd.read_csv(prediction_path)
    required_columns = {"crop_id", "noise_predicted_prob", "noise_predicted_label"}
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"LR prediction CSV missing columns: {sorted(missing_columns)}")
    return predictions


def main(config: dict | None = None) -> None:
    if config is None:
        config = utils.load_pipeline_config()

    utils.init_db(config=config)
    db_path = utils.join_data_root(config["path"]["db_path"], config=config)
    crops_storage_config = config["crops_storage"]
    
    if crops_storage_config["reprocess"]:
        logger.info("crops_storage配置为reprocess=true，将重新裁剪所有入选crop图像")
    else:
        logger.info("crops_storage配置为reprocess=false，将复用已存在的crop图像并只补齐缺失文件")
            
        

    if crops_storage_config["format"] == "flatten":
        with sqlite3.connect(db_path) as conn:
            metadata_columns = list(crops_storage_config["image_metadata_columns"])
            validate_config_column_names(metadata_columns)
            image_select_cols = [f"i.{col}" for col in metadata_columns]
            crop_sql = f"""
                SELECT
                    c.id AS crop_id,
                    c.image_id,
                    c.saved,
                    c.crop_path,
                    c.noise_review_label,
                    c.manual_corrected_label,
                    CASE
                        WHEN c.noise_review_label = '{constants.NOISE_REVIEW_LABEL_OK}' THEN 1
                        ELSE 0
                    END AS manual_reviewed,
                    c.box_x1,
                    c.box_y1,
                    c.box_x2,
                    c.box_y2,
                    {', '.join(image_select_cols)}
                FROM crops c
                JOIN images i ON i.id = c.image_id
                ORDER BY c.id
            """
            metadata = pd.read_sql_query(crop_sql, conn)
        logger.info("已加载%d条crop及图片元数据，开始应用筛选策略。", len(metadata))

        if crops_storage_config["save_only_manual_reviewed"]:
            logger.info("将仅保存人工审核为ok的crop。")
            metadata = metadata[
                metadata["noise_review_label"].eq(constants.NOISE_REVIEW_LABEL_OK)
            ].copy()
        else:
            logger.info("将保存人工审核过、自动审核过和未审核的入选crop。")

        corrected_mask = (
            metadata["manual_corrected_label"].notna()
            & (metadata["manual_corrected_label"].astype(str).str.strip() != "")
        )
        
        if crops_storage_config["exclude_manual_noise"]:
            before_count = len(metadata)
            review_label = metadata["noise_review_label"].fillna("").astype(str).str.strip()
            #为选定噪声label的数据
            manual_noise = review_label.isin(set(crops_storage_config["manual_noise_labels"]))
            corrected_wrong_label = (
                review_label.eq(constants.NOISE_REVIEW_LABEL_WRONG_LABEL)
                & corrected_mask
            ) #被人工纠正为非clean的旧轮次数据
            
            #去掉manual_noise,但是用并集保留被人工纠正为非clean的旧轮次数据（不管新旧轮次）
            metadata = metadata.loc[~manual_noise | corrected_wrong_label].reset_index(drop=True)
            
            logger.info(
                "按人工复核过滤crop：过滤%d条，保留%d条。",
                before_count - len(metadata),
                len(metadata),
            )

        
        label_column = crops_storage_config["label_column"]
        if label_column not in metadata.columns:
            raise ValueError(f"image metadata中不存在label列: {label_column!r}")
        corrected_mask = (
            metadata["manual_corrected_label"].notna()
            & (metadata["manual_corrected_label"].astype(str).str.strip() != "")
        )
        if corrected_mask.any():
            # 将人工纠正的标签应用到label_column中，覆盖原有标签
            metadata.loc[corrected_mask, label_column] = metadata.loc[
                corrected_mask,
                "manual_corrected_label",
            ].astype(str)
            logger.info("已应用%d条人工纠正标签。", int(corrected_mask.sum()))
            invalidate_columns = list(crops_storage_config["manual_correction_invalidate_metadata_columns"])
            metadata = invalidate_metadata_for_manual_corrections(
                metadata,
                corrected_mask=corrected_mask,
                columns=invalidate_columns,
                label_column=label_column,
            )
            if invalidate_columns:
                logger.info(
                    "已清空%d条人工纠正样本的metadata列: %s",
                    int(corrected_mask.sum()),
                    invalidate_columns,
                )
        #如果配置了exclude_manual_noise，则在应用人工纠正标签后再过滤一次，去掉LR判定noise的数据。
        if crops_storage_config["exclude_predicted_noise"]:
            predictions = load_noise_predictions(config, crops_storage_config)
            before_count = len(metadata)
            corrected_crop_ids = set(metadata.loc[corrected_mask, "crop_id"].astype(int))
            metadata = metadata.merge(
                predictions[["crop_id", "noise_predicted_prob", "noise_predicted_label"]],
                on="crop_id",
                how="left",
            )
            predicted_noise_labels = set(crops_storage_config["predicted_noise_labels"])
            predicted_noise_min_prob = float(crops_storage_config["predicted_noise_min_prob"])
            predicted_noise_mask = (
                metadata["noise_predicted_label"].isin(predicted_noise_labels)
                & (metadata["noise_predicted_prob"].fillna(0.0) >= predicted_noise_min_prob)
                & ~metadata["crop_id"].astype(int).isin(corrected_crop_ids)
            )
            #去掉LR过滤
            metadata = metadata.loc[~predicted_noise_mask].reset_index(drop=True)
            logger.info(
                "按LR预测过滤crop：过滤%d条，保留%d条。",
                before_count - len(metadata),
                len(metadata),
            )
        #按格式变换ASCII label
        metadata = build_en_label(
            metadata,
            label_column=label_column,
        )
        
        logger.info("已加载%d条待裁剪crop及图片元数据。", len(metadata))
        logger.info("metadata列: %s", list(metadata.columns))
        
        dataset_root = utils.join_data_root(config['path']["dataset_dir"], config=config)
        dataset_img_subdir = config['path']["dataset_img_subdir"]
        image_extension = crops_storage_config["image_extension"].lower().lstrip(".")
        output_filenames = (
            metadata["image_id"].map(lambda value: f"{int(value):08d}")
            + "_"
            + metadata["crop_id"].map(lambda value: f"{int(value):08d}")
            + f".{image_extension}"
        )
        metadata["image_path"] = dataset_img_subdir + "/" + output_filenames
        metadata["output_path"] = metadata["image_path"].map(lambda path: dataset_root / path) # type: ignore
        metadata["source_path"] = metadata["downloaded_path"].map(
            lambda path: utils.join_data_root(str(path), config=config)
        )

        if crops_storage_config["reprocess"]:
            reusable_mask = pd.Series(False, index=metadata.index)
        else:
            reusable_mask = (
                metadata["crop_path"].notna()
                & (metadata["crop_path"].astype(str).str.strip() == metadata["image_path"].astype(str))
                & metadata["output_path"].map(lambda path: Path(path).exists())
            )
        reusable_count = int(reusable_mask.sum())
        to_save = metadata.loc[~reusable_mask].copy()
        reusable_rows = metadata.loc[reusable_mask].copy()
        logger.info(
            "crop图像复用%d条，需要裁剪保存%d条。",
            reusable_count,
            len(to_save),
        )

        saved_rows = reusable_rows.to_dict(orient="records")
        db_updates = []
        db_update_batch_size = 100
        for _, row in tqdm(to_save.iterrows(), total=len(to_save), desc="存盘裁剪图像"):
            try:
                save_crop_image(
                    source_image_path=row["source_path"],
                    output_path=row["output_path"],
                    box_x1=row["box_x1"],
                    box_y1=row["box_y1"],
                    box_x2=row["box_x2"],
                    box_y2=row["box_y2"],
                    pad_frac=crops_storage_config["crop_pad_frac"],
                    image_format=image_extension,
                    jpeg_quality=crops_storage_config['jpeg_quality']
                )
                saved_row = row.to_dict()
                saved_rows.append(saved_row)
                db_updates.append((row["image_path"], int(row["crop_id"])))
                if len(db_updates) >= db_update_batch_size:
                    flush_crop_save_updates(db_path, db_updates)
                    db_updates.clear()
            except Exception as e:
                logger.error(f"保存crop_id={row['crop_id']}失败: {e}")
                continue

        flush_crop_save_updates(db_path, db_updates)
        
        metadata = pd.DataFrame(saved_rows)
        if metadata.empty:
            logger.warning("没有成功保存的crop，跳过metadata和labels写入。")
            return constants.STAGE_PASS  # type: ignore
        
        metadata, labels = build_label_tables(metadata)
        output_metadata_columns = list(crops_storage_config["metadata_columns"])
        missing_metadata_columns = [col for col in output_metadata_columns if col not in metadata.columns]
        if missing_metadata_columns:
            raise ValueError(f"metadata输出列不存在: {missing_metadata_columns}")
        metadata = metadata[output_metadata_columns]

        metadata_path = dataset_root / crops_storage_config["metadata_file_name"]
        labels_path = dataset_root / crops_storage_config["labels_file_name"]
        manifest_path = dataset_root / "manifest.json"
        dataset_root.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(metadata_path, index=False, encoding="utf-8")
        labels.to_csv(labels_path, index=False, encoding="utf-8")
        write_dataset_manifest(
            manifest_path=manifest_path,
            metadata=metadata,
            labels=labels,
            crops_storage_config=crops_storage_config,
        )

        logger.info("crop图像保存完成，已成功保存%d条crop数据。", len(metadata))
        logger.info("metadata已保存至%s，labels已保存至%s，manifest已保存至%s。", metadata_path, labels_path, manifest_path)
        
        return constants.STAGE_COMPLETED #type:ignore flatten格式处理完成，退出程序


if __name__ == "__main__":
    main()
