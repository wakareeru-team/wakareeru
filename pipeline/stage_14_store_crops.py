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


def _clean_metadata_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def refill_unique_metadata_for_manual_corrections(
    metadata: pd.DataFrame,
    *,
    corrected_mask: pd.Series,
    label_column: str,
    operator_columns: list[str],
    submodel_bandai_columns: list[str],
) -> pd.DataFrame:
    validate_config_column_names(operator_columns)
    validate_config_column_names(submodel_bandai_columns)
    refill_columns = [*operator_columns, *submodel_bandai_columns]
    if label_column in refill_columns:
        raise ValueError("人工纠正metadata补齐列不能包含当前label列。")
    missing_columns = [col for col in [label_column, *refill_columns] if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"人工纠正metadata补齐所需列不存在: {missing_columns}")
    if len(submodel_bandai_columns) != 2:
        raise ValueError("crops_storage.manual_correction_refill_submodel_bandai_columns 必须包含两个列名。")

    metadata = metadata.copy()
    reference = metadata.loc[~corrected_mask].copy()
    if reference.empty or not corrected_mask.any():
        return metadata

    labels = metadata.loc[corrected_mask, label_column].dropna().astype(str).str.strip().unique()
    for label in labels:
        if not label:
            continue
        same_label = reference[reference[label_column].astype(str).str.strip() == label]
        if same_label.empty:
            continue
        target = corrected_mask & metadata[label_column].astype(str).str.strip().eq(label)

        for col in operator_columns:
            values = sorted({_clean_metadata_value(value) for value in same_label[col] if _clean_metadata_value(value)})
            if len(values) == 1:
                metadata.loc[target, col] = values[0]

        pair_cols = submodel_bandai_columns
        pairs = {
            tuple(_clean_metadata_value(row[col]) for col in pair_cols)
            for _, row in same_label[pair_cols].iterrows()
        }
        pairs = {pair for pair in pairs if any(pair)}
        if len(pairs) == 1:
            pair = next(iter(pairs))
            for col, value in zip(pair_cols, pair, strict=True):
                metadata.loc[target, col] = value or pd.NA

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


def _clean_l10n_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _most_common_text(values: pd.Series) -> str:
    cleaned = values.map(_clean_l10n_text)
    cleaned = cleaned[cleaned.ne("")]
    if cleaned.empty:
        return ""
    counts = cleaned.value_counts()
    return sorted(counts[counts.eq(counts.max())].index)[0]


def _build_operator_translations(label_metadata: pd.DataFrame) -> dict[str, list[str]]:
    operators: list[tuple[str, str]] = []
    for operator_ja, rows in label_metadata.groupby("operator_jp", sort=False, dropna=True):
        operator_ja = _clean_l10n_text(operator_ja)
        if not operator_ja:
            continue
        operators.append((operator_ja, _most_common_text(rows["operator_en"])))
    operators.sort(key=lambda pair: pair[0])
    return {
        "ja": [operator_ja for operator_ja, _ in operators],
        "en": [operator_en for _, operator_en in operators],
        "zh": ["" for _ in operators],
    }


def _validate_existing_l10n_metadata(existing: object) -> list[dict]:
    if not isinstance(existing, list):
        raise ValueError("既有l10n metadata必须是JSON列表，拒绝覆盖。")

    seen_labels: set[str] = set()
    for index, item in enumerate(existing):
        if not isinstance(item, dict):
            raise ValueError(f"既有l10n metadata第{index}项不是字典，拒绝覆盖。")
        label = item.get("label")
        operator = item.get("operator")
        if not isinstance(label, dict) or not isinstance(operator, dict):
            raise ValueError(f"既有l10n metadata第{index}项缺少label或operator字典，拒绝覆盖。")
        if isinstance(item.get("id"), bool) or not isinstance(item.get("id"), int):
            raise ValueError(f"既有l10n metadata第{index}项的id不是整数，拒绝覆盖。")
        label_ja = label.get("ja")
        if not isinstance(label_ja, str) or not label_ja.strip():
            raise ValueError(f"既有l10n metadata第{index}项缺少label.ja，拒绝覆盖。")
        if any(not isinstance(label.get(language), str) for language in ("en", "zh")):
            raise ValueError(f"既有l10n metadata第{index}项的label翻译不是字符串，拒绝覆盖。")
        if label_ja in seen_labels:
            raise ValueError(f"既有l10n metadata包含重复label.ja={label_ja!r}，拒绝覆盖。")
        seen_labels.add(label_ja)
        for language in ("ja", "en", "zh"):
            if not isinstance(operator.get(language), list):
                raise ValueError(
                    f"既有l10n metadata的operator.{language}必须是列表，拒绝覆盖。"
                )
            if any(not isinstance(value, str) for value in operator[language]):
                raise ValueError(
                    f"既有l10n metadata的operator.{language}元素必须是字符串，拒绝覆盖。"
                )
        operator_lengths = {len(operator[language]) for language in ("ja", "en", "zh")}
        if len(operator_lengths) != 1:
            raise ValueError("既有l10n metadata的operator语言列表长度不一致，拒绝覆盖。")
        if len(operator["ja"]) != len(set(operator["ja"])):
            raise ValueError("既有l10n metadata包含重复operator.ja，拒绝覆盖。")
    return existing


def _preserve_existing_translations(current: dict, existing: dict) -> None:
    current["label"]["en"] = _clean_l10n_text(existing["label"].get("en"))
    current["label"]["zh"] = _clean_l10n_text(existing["label"].get("zh"))

    existing_operator = existing["operator"]
    existing_by_ja = {
        operator_ja: (existing_operator["en"][index], existing_operator["zh"][index])
        for index, operator_ja in enumerate(existing_operator["ja"])
    }
    for index, operator_ja in enumerate(current["operator"]["ja"]):
        if operator_ja not in existing_by_ja:
            continue
        operator_en, operator_zh = existing_by_ja[operator_ja]
        current["operator"]["en"][index] = _clean_l10n_text(operator_en)
        current["operator"]["zh"][index] = _clean_l10n_text(operator_zh)


def build_l10n_metadata(
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    existing_path: Path,
) -> tuple[list[dict], int]:
    required_columns = {"label", "operator_jp", "operator_en", "wiki_title"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"生成l10n metadata缺少metadata列: {sorted(missing_columns)}")

    existing_by_label: dict[str, dict] = {}
    if existing_path.exists():
        with existing_path.open("r", encoding="utf-8") as file:
            existing = _validate_existing_l10n_metadata(json.load(file))
        existing_by_label = {item["label"]["ja"]: item for item in existing}

    result = []
    preserved_count = 0
    for row in labels.itertuples(index=False):
        label_metadata = metadata.loc[metadata["label"].eq(row.label)]
        item = {
            "id": int(row.label_id),
            "label": {"ja": row.label, "en": "", "zh": ""},
            "operator": _build_operator_translations(label_metadata),
            "wiki_title_ja": _most_common_text(label_metadata["wiki_title"]),
        }
        existing = existing_by_label.get(row.label)
        if existing is not None and existing.get("id") == int(row.label_id):
            _preserve_existing_translations(item, existing)
            preserved_count += 1
        result.append(item)
    return result, preserved_count


def write_l10n_metadata(labels: pd.DataFrame, metadata: pd.DataFrame, output_path: Path) -> int:
    l10n_metadata, preserved_count = build_l10n_metadata(labels, metadata, output_path)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(l10n_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return preserved_count


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
        "l10n_metadata_file": crops_storage_config["l10n_metadata_file_name"],
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
            "manual_correction_refill_operator_columns": crops_storage_config[
                "manual_correction_refill_operator_columns"
            ],
            "manual_correction_refill_submodel_bandai_columns": crops_storage_config[
                "manual_correction_refill_submodel_bandai_columns"
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
            metadata = refill_unique_metadata_for_manual_corrections(
                metadata,
                corrected_mask=corrected_mask,
                label_column=label_column,
                operator_columns=list(crops_storage_config["manual_correction_refill_operator_columns"]),
                submodel_bandai_columns=list(
                    crops_storage_config["manual_correction_refill_submodel_bandai_columns"]
                ),
            )
            logger.info(
                "已按唯一反查规则尝试补齐%d条人工纠正样本的operator与submodel/bandai。",
                int(corrected_mask.sum()),
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
        l10n_source_metadata = metadata
        output_metadata_columns = list(crops_storage_config["metadata_columns"])
        missing_metadata_columns = [col for col in output_metadata_columns if col not in metadata.columns]
        if missing_metadata_columns:
            raise ValueError(f"metadata输出列不存在: {missing_metadata_columns}")
        metadata = metadata[output_metadata_columns]

        metadata_path = dataset_root / crops_storage_config["metadata_file_name"]
        labels_path = dataset_root / crops_storage_config["labels_file_name"]
        l10n_metadata_path = dataset_root / crops_storage_config["l10n_metadata_file_name"]
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
        preserved_translation_count = write_l10n_metadata(
            labels,
            l10n_source_metadata,
            l10n_metadata_path,
        )
        logger.info(
            "多语言metadata已保存至%s；按label.ja和id保留%d条既有翻译。",
            l10n_metadata_path,
            preserved_translation_count,
        )

        return constants.STAGE_COMPLETED #type:ignore flatten格式处理完成，退出程序


if __name__ == "__main__":
    main()
