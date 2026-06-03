from __future__ import annotations

import argparse
import base64
import html
import re
import sqlite3
import sys
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from PIL import Image, ImageDraw


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError(f"Project root not found from {start}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import constants, utils  # noqa: E402


LABEL_REVIEW_CONFIG = {
    "label_granularity": "config",
    "crop_status": "all",
    "loss_round": "latest",
    "sample_mode": "lr_auto_filtered",
    "samples_per_label": 3,
    "sample_size": 120,
    "random_seed": 42,
    "gallery_samples_per_label": 8,
    "crop_pad_frac": 0.04,
    "stats_top_n": 80,
    "skip_reviewed": False,
    "review_score_col": "label_review",
}

CONFIG: dict[str, Any] = {}
DB_PATH: Path


def quote_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQLite identifier: {name!r}")
    return f'"{name}"'


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({quote_ident(table_name)})")}


def crop_columns(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("PRAGMA table_info(crops)").fetchall()
    return {row[1]: row[2] or "" for row in rows}


def ensure_review_columns() -> None:
    column_defs = {
        "noise_review_label": "TEXT",
        "noise_review_note": "TEXT",
        "noise_reviewed_at": "TEXT",
        "noise_review_score_col": "TEXT",
        "manual_corrected_label": "TEXT",
        "manual_corrected_at": "TEXT",
    }
    with sqlite3.connect(DB_PATH) as conn:
        cols = crop_columns(conn)
        for col, col_type in column_defs.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE crops ADD COLUMN {quote_ident(col)} {col_type}")
        conn.commit()


def label_expr_for_granularity(label_granularity: str) -> str:
    if label_granularity == "config":
        label_granularity = CONFIG["noise_detection"]["label_granularity"]
    if label_granularity == "submodel":
        return "COALESCE(NULLIF(i.submodel, ''), NULLIF(i.fine_grained_series, ''), c.series)"
    if label_granularity == "fine_grained_series":
        return "COALESCE(NULLIF(i.fine_grained_series, ''), c.series)"
    if label_granularity == "series":
        return "c.series"
    raise ValueError("label_granularity must be one of: config, series, fine_grained_series, submodel")


def metadata_sql(label_granularity: str, crop_status: str) -> str:
    label_expr = label_expr_for_granularity(label_granularity)
    status_where = ""
    if crop_status != "all":
        if crop_status not in constants.CROP_STATUSES:
            raise ValueError(f"crop_status must be one of: all, {sorted(constants.CROP_STATUSES)}")
        status_where = f"          AND c.crop_status = '{crop_status}'\n"
    optional_crop_cols = {
        "noise_review_label": "c.noise_review_label AS noise_review_label",
        "noise_review_note": "c.noise_review_note AS noise_review_note",
        "noise_reviewed_at": "c.noise_reviewed_at AS noise_reviewed_at",
        "manual_corrected_label": "c.manual_corrected_label AS manual_corrected_label",
        "manual_corrected_at": "c.manual_corrected_at AS manual_corrected_at",
        "noise_predicted_label": "c.noise_predicted_label AS noise_predicted_label",
        "noise_predicted_prob": "c.noise_predicted_prob AS noise_predicted_prob",
    }
    optional_image_cols = {
        "fine_grained_series": "i.fine_grained_series AS fine_grained_series",
        "submodel": "i.submodel AS submodel",
        "bandai": "i.bandai AS bandai",
        "operator_jp": "i.operator_jp AS operator_jp",
        "operator_en": "i.operator_en AS operator_en",
        "special_formation": "i.special_formation AS special_formation",
        "special_livery": "i.special_livery AS special_livery",
    }

    with sqlite3.connect(DB_PATH) as conn:
        crop_cols = table_columns(conn, "crops")
        image_cols = table_columns(conn, "images")

    optional_selects = [
        expr for col, expr in optional_crop_cols.items() if col in crop_cols
    ] + [
        expr for col, expr in optional_image_cols.items() if col in image_cols
    ]
    optional_sql = ",\n            " + ",\n            ".join(optional_selects) if optional_selects else ""

    return f"""
        SELECT
            c.id AS crop_id,
            c.image_id,
            c.series,
            c.power_type,
            c.crop_index,
            c.detector_label,
            c.detector_score,
            c.box_x1,
            c.box_y1,
            c.box_x2,
            c.box_y2,
            c.box_area,
            c.crop_status,
            c.crop_reason,
            i.file_title,
            i.downloaded_path,
            i.category,
            {label_expr} AS label{optional_sql}
        FROM crops c
        JOIN images i ON i.id = c.image_id
        WHERE i.downloaded_path IS NOT NULL
          AND TRIM(i.downloaded_path) != ''
{status_where.rstrip()}
        ORDER BY c.id
    """


def resolve_loss_round(loss_round: str) -> tuple[str, Path]:
    active_round = str(loss_round).strip() or "latest"
    round_dir = utils.get_loss_round_dir(config=CONFIG, active_round=active_round)
    return round_dir.name, round_dir


def load_loss_features(loss_round: str) -> pd.DataFrame:
    _, round_dir = resolve_loss_round(loss_round)
    feature_path = round_dir / CONFIG["loss_analysis"]["loss_feature_file_name"]
    if not feature_path.exists():
        return pd.DataFrame()
    features = pd.read_csv(feature_path)
    if "crop_id" not in features.columns:
        raise ValueError(f"Loss feature CSV missing required column: {feature_path}")
    features["crop_id"] = features["crop_id"].astype(int)
    keep_cols = [
        "crop_id",
        "mean",
        "loss_tail_mean",
        "loss_mean_pct_in_label",
        "error_rate",
        "pred_label",
        "pred_label_rate",
        "noise_score_v1",
    ]
    return features[[col for col in keep_cols if col in features.columns]].copy()


def load_lr_prediction_file(loss_round: str) -> pd.DataFrame:
    _, round_dir = resolve_loss_round(loss_round)
    prediction_path = round_dir / CONFIG["lr_prediction"]["prediction_file_name"]
    if not prediction_path.exists():
        return pd.DataFrame()
    predictions = pd.read_csv(prediction_path)
    if "crop_id" not in predictions.columns:
        raise ValueError(f"LR prediction CSV missing required column: {prediction_path}")
    keep_cols = [
        "crop_id",
        "mean",
        "loss_tail_mean",
        "loss_mean_pct_in_label",
        "error_rate",
        "pred_label",
        "pred_label_rate",
        "noise_score_v1",
        "noise_predicted_prob",
        "noise_predicted_label",
        "noise_prediction_model",
        "predicted_at",
    ]
    predictions = predictions[[col for col in keep_cols if col in predictions.columns]].copy()
    predictions["crop_id"] = predictions["crop_id"].astype(int)
    return predictions.drop_duplicates("crop_id", keep="last")


def load_rows(label_granularity: str, crop_status: str, loss_round: str) -> pd.DataFrame:
    ensure_review_columns()
    with sqlite3.connect(DB_PATH) as conn:
        rows = pd.read_sql_query(metadata_sql(label_granularity, crop_status), conn)
    rows["label"] = rows["label"].fillna(rows["series"]).astype(str)
    rows["crop_id"] = rows["crop_id"].astype(int)
    features = load_loss_features(loss_round)
    if not features.empty:
        rows = rows.merge(features, on="crop_id", how="left")
    predictions = load_lr_prediction_file(loss_round)
    if not predictions.empty:
        overlap_cols = sorted((set(rows.columns) & set(predictions.columns)) - {"crop_id", "label"})
        renamed_predictions = predictions.rename(
            columns={col: f"prediction_{col}" for col in overlap_cols}
        )
        rows = rows.merge(renamed_predictions, on="crop_id", how="left")
        for col in overlap_cols:
            prediction_col = f"prediction_{col}"
            rows[col] = rows[prediction_col].fillna(rows[col])
    if "noise_predicted_prob" in rows.columns:
        rows["noise_predicted_prob"] = pd.to_numeric(rows["noise_predicted_prob"], errors="coerce")
    if "manual_corrected_label" in rows.columns:
        corrected = rows["manual_corrected_label"].fillna("").astype(str).str.strip()
        rows["effective_label"] = rows["label"].where(corrected == "", corrected)
    else:
        rows["effective_label"] = rows["label"]
    return rows


def known_label_choices(rows: pd.DataFrame) -> list[str]:
    labels = set(rows["label"].dropna().astype(str))
    if "manual_corrected_label" in rows.columns:
        corrected = rows["manual_corrected_label"].dropna().astype(str).str.strip()
        labels.update(label for label in corrected if label)
    if "pred_label" in rows.columns:
        predicted = rows["pred_label"].dropna().astype(str).str.strip()
        labels.update(label for label in predicted if label)
    return sorted(labels)


def filter_rows(
    rows: pd.DataFrame,
    label_query: str,
    skip_reviewed: bool,
    only_uncorrected: bool,
) -> pd.DataFrame:
    filtered = rows.copy()
    label_query = str(label_query or "").strip()
    if label_query:
        mask = pd.Series(False, index=filtered.index)
        for col in ["label", "effective_label", "series", "fine_grained_series", "submodel", "category"]:
            if col in filtered.columns:
                mask = mask | filtered[col].astype(str).str.contains(label_query, case=False, regex=False, na=False)
        filtered = filtered[mask].copy()
    if skip_reviewed and "noise_review_label" in filtered.columns:
        reviewed = filtered["noise_review_label"].fillna("").astype(str).str.strip() != ""
        filtered = filtered[~reviewed].copy()
    if only_uncorrected and "manual_corrected_label" in filtered.columns:
        corrected = filtered["manual_corrected_label"].fillna("").astype(str).str.strip() != ""
        filtered = filtered[~corrected].copy()
    return filtered


def label_count_table(rows: pd.DataFrame, top_n: int | None = None) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "label",
                "count",
                "percent",
                "ratio_to_median",
                "series_count",
                "reviewed_count",
                "corrected_count",
            ]
        )

    grouped = rows.groupby("label", dropna=False)
    counts = grouped.size().rename("count").reset_index()
    counts["label"] = counts["label"].fillna("<missing>").astype(str)
    counts["percent"] = counts["count"] / max(1, len(rows))
    median_count = max(1.0, float(counts["count"].median()))
    counts["ratio_to_median"] = counts["count"] / median_count
    counts["series_count"] = grouped["series"].nunique().to_numpy()
    if "noise_review_label" in rows.columns:
        counts["reviewed_count"] = grouped["noise_review_label"].apply(
            lambda values: int(values.fillna("").astype(str).str.strip().ne("").sum())
        ).to_numpy()
    else:
        counts["reviewed_count"] = 0
    if "manual_corrected_label" in rows.columns:
        counts["corrected_count"] = grouped["manual_corrected_label"].apply(
            lambda values: int(values.fillna("").astype(str).str.strip().ne("").sum())
        ).to_numpy()
    else:
        counts["corrected_count"] = 0
    counts = counts.sort_values(["count", "label"], ascending=[False, True]).reset_index(drop=True)
    counts["cumulative_percent"] = counts["count"].cumsum() / max(1, len(rows))
    if top_n is not None:
        counts = counts.head(max(1, int(top_n))).copy()
    return counts


def sample_rows(
    rows: pd.DataFrame,
    stats: pd.DataFrame,
    sample_mode: str,
    samples_per_label: int,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if rows.empty:
        return rows

    samples_per_label = max(1, int(samples_per_label))
    sample_size = max(1, int(sample_size))
    seed = int(seed)

    def sample_per_label(work: pd.DataFrame) -> pd.DataFrame:
        parts = [
            group.sample(n=min(samples_per_label, len(group)), random_state=seed)
            for _, group in work.groupby("label", sort=False)
        ]
        if not parts:
            return work.head(0).copy()
        return pd.concat(parts, ignore_index=True)

    if sample_mode == "label_balanced_random":
        sample = sample_per_label(rows)
        return sample.sample(n=min(sample_size, len(sample)), random_state=seed).reset_index(drop=True)

    if sample_mode == "rare_labels":
        rare_labels = stats.sort_values(["count", "label"], ascending=[True, True])["label"].head(sample_size).tolist()
        work = rows[rows["label"].isin(rare_labels)].copy()
        return sample_per_label(work).head(sample_size)

    if sample_mode == "large_labels":
        large_labels = stats.sort_values(["count", "label"], ascending=[False, True])["label"].head(sample_size).tolist()
        work = rows[rows["label"].isin(large_labels)].copy()
        return sample_per_label(work).head(sample_size)

    if sample_mode == "proportional_random":
        return rows.sample(n=min(sample_size, len(rows)), random_state=seed).reset_index(drop=True)

    if sample_mode == "lr_auto_filtered":
        required_cols = {"noise_predicted_label", "noise_predicted_prob"}
        if not required_cols.issubset(rows.columns):
            raise gr.Error("当前数据没有 LR 预测列。请先运行 stage 13，或确认当前 loss round 有 lr_predictions.csv。")
        predicted_labels = set(CONFIG["crops_storage"]["predicted_noise_labels"])
        min_prob = float(CONFIG["crops_storage"]["predicted_noise_min_prob"])
        review_label = rows.get("noise_review_label", pd.Series("", index=rows.index)).fillna("").astype(str).str.strip()
        corrected = rows.get("manual_corrected_label", pd.Series("", index=rows.index)).fillna("").astype(str).str.strip()
        work = rows[
            review_label.eq("")
            & corrected.eq("")
            & rows["noise_predicted_label"].isin(predicted_labels)
            & (rows["noise_predicted_prob"].fillna(0.0) >= min_prob)
        ].copy()
        if work.empty:
            return work
        if "pred_label" in work.columns:
            work["has_pred_label"] = work["pred_label"].notna() & work["pred_label"].astype(str).str.strip().ne("")
            return (
                work.sort_values(["has_pred_label", "noise_predicted_prob"], ascending=[False, False])
                .drop(columns=["has_pred_label"])
                .head(sample_size)
                .reset_index(drop=True)
            )
        return (
            work.sort_values("noise_predicted_prob", ascending=False)
            .head(sample_size)
            .reset_index(drop=True)
        )

    if sample_mode == "corrected":
        if "manual_corrected_label" not in rows.columns:
            return rows.head(0)
        corrected = rows[rows["manual_corrected_label"].fillna("").astype(str).str.strip() != ""].copy()
        return corrected.sample(n=min(sample_size, len(corrected)), random_state=seed).reset_index(drop=True)

    raise gr.Error(f"未知抽样方式: {sample_mode}")


def placeholder_image(message: str, size: tuple[int, int] = (512, 384)) -> Image.Image:
    img = Image.new("RGB", size, "#f2f2f2")
    draw = ImageDraw.Draw(img)
    draw.multiline_text((20, 20), message, fill="#333333")
    return img


def load_crop_image(row: dict[str, Any], pad_frac: float) -> Image.Image:
    try:
        return utils.load_crop(row, config=CONFIG, pad_frac=float(pad_frac))
    except Exception as exc:
        return placeholder_image(f"读取图片失败 crop_id={row.get('crop_id')}\n{exc}")


def caption_for_row(row: dict[str, Any]) -> str:
    parts = [f"crop_id={row.get('crop_id')}", f"label={row.get('label')}"]
    corrected = str(row.get("manual_corrected_label") or "").strip()
    if corrected:
        parts.append(f"corrected={corrected}")
    if row.get("series"):
        parts.append(f"series={row.get('series')}")
    if row.get("detector_score") is not None and row.get("detector_score") == row.get("detector_score"):
        parts.append(f"det={float(row['detector_score']):.3f}")
    if row.get("noise_predicted_prob") is not None and row.get("noise_predicted_prob") == row.get("noise_predicted_prob"):
        parts.append(f"LR={float(row['noise_predicted_prob']):.3f}")
    pred_label = row.get("pred_label")
    if pred_label is not None and pred_label == pred_label and str(pred_label).strip():
        parts.append(f"pred={row.get('pred_label')}")
    return " | ".join(parts)


def fit_thumbnail(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    thumb = img.copy()
    thumb.thumbnail(size)
    canvas = Image.new("RGB", size, "#f7f7f7")
    left = (size[0] - thumb.width) // 2
    top = (size[1] - thumb.height) // 2
    canvas.paste(thumb, (left, top))
    return canvas


def image_to_data_uri(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def make_label_gallery_html(
    sample: pd.DataFrame,
    samples_per_label: int,
    pad_frac: float,
) -> str:
    if sample.empty:
        return "<div class='label-gallery-empty'>没有抽样结果。</div>"

    sections = [
        """
        <style>
        .wak-label-gallery {
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .wak-label-group {
            border: 1px solid #d9d9d9;
            border-radius: 8px;
            background: #ffffff;
            overflow: hidden;
        }
        .wak-label-header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 9px 12px;
            border-bottom: 1px solid #e7e7e7;
            background: #f7f7f7;
            font-weight: 600;
        }
        .wak-label-pill {
            display: inline-flex;
            align-items: center;
            min-height: 24px;
            padding: 2px 9px;
            border: 1px solid #c9c9c9;
            border-radius: 999px;
            background: #ffffff;
            color: #111111;
            font-size: 13px;
        }
        .wak-label-count {
            color: #555555;
            font-size: 12px;
            font-weight: 400;
        }
        .wak-label-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            padding: 12px;
        }
        .wak-crop-card {
            width: 156px;
        }
        .wak-crop-card img {
            width: 156px;
            height: 116px;
            object-fit: contain;
            border: 1px solid #dddddd;
            border-radius: 6px;
            background: #f7f7f7;
        }
        .wak-crop-caption {
            margin-top: 4px;
            color: #555555;
            font-size: 11px;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        .label-gallery-empty {
            padding: 16px;
            color: #555555;
        }
        </style>
        <div class="wak-label-gallery">
        """
    ]

    for label, group in sample.groupby("label", sort=True):
        rows = group.to_dict(orient="records")
        label_text = html.escape(str(label))
        sections.append(
            f"""
            <section class="wak-label-group">
                <div class="wak-label-header">
                    <span class="wak-label-pill">{label_text}</span>
                    <span class="wak-label-count">{len(rows)} sampled</span>
                </div>
                <div class="wak-label-grid">
            """
        )
        for row in rows[: max(1, int(samples_per_label))]:
            thumb = fit_thumbnail(load_crop_image(row, pad_frac), (156, 116))
            caption = html.escape(caption_for_row(row))
            sections.append(
                f"""
                <figure class="wak-crop-card">
                    <img src="{image_to_data_uri(thumb)}" alt="{caption}">
                    <figcaption class="wak-crop-caption">{caption}</figcaption>
                </figure>
                """
            )
        sections.append("</div></section>")

    sections.append("</div>")
    return "\n".join(sections)


def preview_columns(sample: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "crop_id",
        "label",
        "effective_label",
        "series",
        "fine_grained_series",
        "submodel",
        "bandai",
        "operator_jp",
        "operator_en",
        "special_formation",
        "special_livery",
        "detector_score",
        "noise_review_label",
        "manual_corrected_label",
        "noise_predicted_prob",
        "noise_predicted_label",
        "pred_label",
        "pred_label_rate",
        "error_rate",
        "file_title",
        "category",
        "downloaded_path",
    ]
    return sample[[col for col in columns if col in sample.columns]].copy()


def summary_markdown(rows: pd.DataFrame, filtered: pd.DataFrame, sample: pd.DataFrame, stats: pd.DataFrame) -> str:
    reviewed = (
        int(filtered["noise_review_label"].fillna("").astype(str).str.strip().ne("").sum())
        if "noise_review_label" in filtered.columns and not filtered.empty
        else 0
    )
    corrected = (
        int(filtered["manual_corrected_label"].fillna("").astype(str).str.strip().ne("").sum())
        if "manual_corrected_label" in filtered.columns and not filtered.empty
        else 0
    )
    max_count = int(stats["count"].max()) if not stats.empty else 0
    min_count = int(stats["count"].min()) if not stats.empty else 0
    median_count = float(stats["count"].median()) if not stats.empty else 0.0
    lines = [
        "### Label Review 概览",
        f"- **全部 crop 数**: {len(rows)}",
        f"- **筛选后 crop 数**: {len(filtered)}",
        f"- **筛选后 label 数**: {filtered['label'].nunique() if not filtered.empty else 0}",
        f"- **抽样 crop 数**: {len(sample)}",
        f"- **抽样覆盖 label 数**: {sample['label'].nunique() if not sample.empty else 0}",
        f"- **已复核 / 已修正**: {reviewed} / {corrected}",
        f"- **label count min / median / max**: {min_count} / {median_count:.1f} / {max_count}",
    ]
    if {"noise_predicted_label", "noise_predicted_prob"}.issubset(filtered.columns):
        predicted_labels = set(CONFIG["crops_storage"]["predicted_noise_labels"])
        min_prob = float(CONFIG["crops_storage"]["predicted_noise_min_prob"])
        auto_filtered = (
            filtered["noise_predicted_label"].isin(predicted_labels)
            & (filtered["noise_predicted_prob"].fillna(0.0) >= min_prob)
        )
        lines.append(f"- **LR 自动过滤候选**: {int(auto_filtered.sum())} @ prob>={min_prob:g}")
    return "\n".join(lines)


def display_record(records: list[dict[str, Any]], idx: int, pad_frac: float):
    total = len(records or [])
    if total == 0:
        return placeholder_image("没有抽样结果。"), "没有抽样结果。", None, "", "", "0/0"
    idx = max(0, min(int(idx), total - 1))
    row = records[idx]
    fields = [
        ("progress", f"{idx + 1}/{total}"),
        ("crop_id", row.get("crop_id")),
        ("label", row.get("label")),
        ("effective_label", row.get("effective_label")),
        ("series", row.get("series")),
        ("fine_grained_series", row.get("fine_grained_series")),
        ("submodel", row.get("submodel")),
        ("bandai", row.get("bandai")),
        ("operator", row.get("operator_jp") or row.get("operator_en")),
        ("formation", row.get("special_formation")),
        ("livery", row.get("special_livery")),
        ("detector_score", f"{float(row.get('detector_score')):.4f}" if pd.notna(row.get("detector_score")) else None),
        ("LR_predicted_label", row.get("noise_predicted_label")),
        ("LR_predicted_prob", f"{float(row.get('noise_predicted_prob')):.4f}" if pd.notna(row.get("noise_predicted_prob")) else None),
        ("linear_head_pred_label", row.get("pred_label")),
        ("linear_head_pred_label_rate", f"{float(row.get('pred_label_rate')):.4f}" if pd.notna(row.get("pred_label_rate")) else None),
        ("error_rate", f"{float(row.get('error_rate')):.4f}" if pd.notna(row.get("error_rate")) else None),
        ("review_label", row.get("noise_review_label")),
        ("manual_corrected_label", row.get("manual_corrected_label")),
        ("file_title", row.get("file_title")),
        ("category", row.get("category")),
    ]
    lines = ["### 当前样本"]
    for key, value in fields:
        if value is not None and value == value and str(value).strip() != "":
            lines.append(f"- **{key}**: {value}")
    note = row.get("noise_review_note") or ""
    corrected = row.get("manual_corrected_label") or ""
    return load_crop_image(row, pad_frac), "\n".join(lines), row.get("noise_review_label"), note, corrected, f"{idx + 1}/{total}"


def save_label_review(
    crop_id: int,
    review_label: str,
    note: str,
    corrected_label: str,
) -> None:
    review_label = str(review_label or "").strip()
    corrected_label = str(corrected_label or "").strip()
    if not review_label:
        raise gr.Error("请先选择审核结果。")
    if review_label == constants.NOISE_REVIEW_LABEL_WRONG_LABEL and not corrected_label:
        raise gr.Error("标记为 wrong_label 时需要填写修正后的 label。")
    if review_label != constants.NOISE_REVIEW_LABEL_WRONG_LABEL:
        corrected_label = ""

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE crops
            SET noise_review_label = ?,
                noise_review_note = ?,
                noise_reviewed_at = CURRENT_TIMESTAMP,
                noise_review_score_col = ?,
                manual_corrected_label = ?,
                manual_corrected_at = CASE WHEN ? = '' THEN NULL ELSE CURRENT_TIMESTAMP END
            WHERE id = ?
            """,
            (
                review_label,
                note or None,
                LABEL_REVIEW_CONFIG["review_score_col"],
                corrected_label or None,
                corrected_label,
                int(crop_id),
            ),
        )
        conn.commit()


def export_sample_csv(records: list[dict[str, Any]]) -> str:
    if not records:
        raise gr.Error("当前没有抽样结果可导出。")
    output_dir = Path(tempfile.gettempdir()) / "wakareeru_label_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"label_review_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return str(path)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Label 分布与抽样复核") as app:
        gr.Markdown("# Label 分布与抽样复核")

        records_state = gr.State([])
        idx_state = gr.State(0)
        label_choices_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                label_granularity = gr.Radio(
                    choices=[
                        ("读取 config noise_detection.label_granularity", "config"),
                        ("series", "series"),
                        ("fine_grained_series", "fine_grained_series"),
                        ("submodel", "submodel"),
                    ],
                    value=LABEL_REVIEW_CONFIG["label_granularity"],
                    label="Label 粒度",
                )
                crop_status = gr.Radio(
                    choices=[
                        ("全部 crop", "all"),
                        ("ok", constants.CROP_STATUS_OK),
                        ("pending", constants.CROP_STATUS_PENDING),
                        ("rejected", constants.CROP_STATUS_REJECTED),
                    ],
                    value=LABEL_REVIEW_CONFIG["crop_status"],
                    label="Crop status",
                )
                loss_round = gr.Textbox(
                    value=LABEL_REVIEW_CONFIG["loss_round"],
                    label="Loss round",
                    placeholder="latest or 20260528_214343",
                )
                sample_mode = gr.Radio(
                    choices=[
                        ("LR 自动过滤样本", "lr_auto_filtered"),
                        ("按 label 均衡随机", "label_balanced_random"),
                        ("优先小 label", "rare_labels"),
                        ("优先大 label", "large_labels"),
                        ("按总体比例随机", "proportional_random"),
                        ("只看已修正样本", "corrected"),
                    ],
                    value=LABEL_REVIEW_CONFIG["sample_mode"],
                    label="抽样方式",
                )
                label_query = gr.Textbox(value="", label="标签 / series / category 包含")
                skip_reviewed = gr.Checkbox(value=LABEL_REVIEW_CONFIG["skip_reviewed"], label="跳过已复核 crop")
                only_uncorrected = gr.Checkbox(value=False, label="跳过已有 corrected label 的 crop")
                samples_per_label = gr.Slider(
                    1,
                    30,
                    value=LABEL_REVIEW_CONFIG["samples_per_label"],
                    step=1,
                    label="每个 label 抽样数",
                )
                sample_size = gr.Slider(1, 2000, value=LABEL_REVIEW_CONFIG["sample_size"], step=1, label="抽样总上限")
                seed = gr.Number(value=LABEL_REVIEW_CONFIG["random_seed"], precision=0, label="随机种子")
                gallery_samples_per_label = gr.Slider(
                    1,
                    20,
                    value=LABEL_REVIEW_CONFIG["gallery_samples_per_label"],
                    step=1,
                    label="图库每个 label 最多显示",
                )
                stats_top_n = gr.Slider(5, 200, value=LABEL_REVIEW_CONFIG["stats_top_n"], step=1, label="统计 Top N")
                pad_frac = gr.Slider(0.0, 0.25, value=LABEL_REVIEW_CONFIG["crop_pad_frac"], step=0.01, label="Crop 外扩比例")
                load_btn = gr.Button("加载 / 重新抽样", variant="primary")
                export_btn = gr.Button("导出当前抽样 CSV")
                export_file = gr.File(label="导出的 CSV")

            with gr.Column(scale=3):
                summary = gr.Markdown()
                with gr.Tabs():
                    with gr.Tab("复核"):
                        with gr.Row():
                            with gr.Column(scale=2):
                                image = gr.Image(label="Crop", type="pil", height=520)
                                meta = gr.Markdown()
                            with gr.Column(scale=1):
                                progress = gr.Textbox(label="进度", interactive=False)
                                review_label = gr.Radio(
                                    choices=[
                                        constants.NOISE_REVIEW_LABEL_OK,
                                        constants.NOISE_REVIEW_LABEL_WRONG_LABEL,
                                        constants.NOISE_REVIEW_LABEL_OUT_OF_LABEL_SPACE,
                                        constants.NOISE_REVIEW_LABEL_BAD_CROP,
                                        constants.NOISE_REVIEW_LABEL_AMBIGUOUS,
                                    ],
                                    label="审核结果",
                                )
                                corrected_label = gr.Dropdown(
                                    choices=[],
                                    label="修正后的 label",
                                    allow_custom_value=True,
                                )
                                note = gr.Textbox(label="备注", lines=4)
                                with gr.Row():
                                    prev_btn = gr.Button("Previous")
                                    skip_btn = gr.Button("Skip")
                                with gr.Row():
                                    ok_btn = gr.Button("OK", variant="secondary")
                                    wrong_btn = gr.Button("Wrong Label", variant="secondary")
                                with gr.Row():
                                    use_pred_btn = gr.Button("Use Pred Label", variant="secondary")
                                    wrong_pred_btn = gr.Button("Wrong = Pred & next", variant="primary")
                                save_next_btn = gr.Button("Save & next", variant="primary")
                    with gr.Tab("图库"):
                        gallery = gr.HTML(label="按 label 分组的抽样 crop")
                        sample_table = gr.Dataframe(label="抽样明细", interactive=False, wrap=True)
                    with gr.Tab("Label 分布"):
                        label_bar = gr.BarPlot(
                            label="Label 数量分布",
                            x="count",
                            y="label",
                            title="Label 数量分布",
                            vertical=False,
                            height=720,
                        )
                        label_table = gr.Dataframe(label="Label 数量表", interactive=False, wrap=True)

        def on_load(
            label_granularity_value,
            crop_status_value,
            loss_round_value,
            sample_mode_value,
            label_query_value,
            skip_reviewed_value,
            only_uncorrected_value,
            samples_per_label_value,
            sample_size_value,
            seed_value,
            gallery_samples_per_label_value,
            stats_top_n_value,
            pad_frac_value,
        ):
            rows = load_rows(
                str(label_granularity_value),
                str(crop_status_value),
                str(loss_round_value or "latest"),
            )
            filtered = filter_rows(
                rows,
                str(label_query_value or ""),
                bool(skip_reviewed_value),
                bool(only_uncorrected_value),
            )
            full_stats = label_count_table(filtered)
            stats_preview = label_count_table(filtered, int(stats_top_n_value))
            sample = sample_rows(
                filtered,
                full_stats,
                str(sample_mode_value),
                int(samples_per_label_value),
                int(sample_size_value),
                int(seed_value),
            )
            records = sample.to_dict(orient="records")
            labels = known_label_choices(rows)
            img, md, current_review, current_note, current_corrected, prog = display_record(records, 0, float(pad_frac_value))
            return (
                records,
                0,
                labels,
                summary_markdown(rows, filtered, sample, full_stats),
                img,
                md,
                current_review,
                current_note,
                gr.update(choices=labels, value=current_corrected or None),
                prog,
                make_label_gallery_html(
                    sample,
                    int(gallery_samples_per_label_value),
                    float(pad_frac_value),
                ),
                preview_columns(sample),
                stats_preview,
                stats_preview,
            )

        def show_index(records, idx, pad_frac_value):
            img, md, current_review, current_note, current_corrected, prog = display_record(
                records or [],
                int(idx),
                float(pad_frac_value),
            )
            return img, md, current_review, current_note, current_corrected, prog

        def on_prev(records, idx, pad_frac_value):
            idx = max(0, int(idx) - 1)
            return idx, *show_index(records, idx, pad_frac_value)

        def on_skip(records, idx, pad_frac_value):
            records = records or []
            idx = min(len(records) - 1, int(idx) + 1) if records else 0
            return idx, *show_index(records, idx, pad_frac_value)

        def on_save_next(records, idx, review_label_value, note_value, corrected_label_value, pad_frac_value):
            records = records or []
            if not records:
                raise gr.Error("当前没有抽样结果，请先加载。")
            idx = max(0, min(int(idx), len(records) - 1))
            row = records[idx]
            save_label_review(
                int(row["crop_id"]),
                str(review_label_value or ""),
                str(note_value or ""),
                str(corrected_label_value or ""),
            )
            row["noise_review_label"] = review_label_value
            row["noise_review_note"] = note_value
            row["manual_corrected_label"] = corrected_label_value if review_label_value == constants.NOISE_REVIEW_LABEL_WRONG_LABEL else ""
            idx = min(len(records) - 1, idx + 1)
            return records, idx, *show_index(records, idx, pad_frac_value)

        def quick_save(records, idx, note_value, corrected_label_value, pad_frac_value, review_label_value):
            return on_save_next(records, idx, review_label_value, note_value, corrected_label_value, pad_frac_value)

        def current_pred_label(records, idx):
            records = records or []
            if not records:
                raise gr.Error("当前没有抽样结果，请先加载。")
            idx = max(0, min(int(idx), len(records) - 1))
            raw_pred_label = records[idx].get("pred_label")
            pred_label = "" if raw_pred_label is None or raw_pred_label != raw_pred_label else str(raw_pred_label).strip()
            if not pred_label:
                raise gr.Error("当前样本没有线性头预测 label。请确认当前 loss round 的 loss feature 包含 pred_label。")
            return pred_label

        def wrong_with_pred_label(records, idx, note_value, pad_frac_value):
            pred_label = current_pred_label(records, idx)
            return on_save_next(
                records,
                idx,
                constants.NOISE_REVIEW_LABEL_WRONG_LABEL,
                note_value,
                pred_label,
                pad_frac_value,
            )

        load_btn.click(
            on_load,
            inputs=[
                label_granularity,
                crop_status,
                loss_round,
                sample_mode,
                label_query,
                skip_reviewed,
                only_uncorrected,
                samples_per_label,
                sample_size,
                seed,
                gallery_samples_per_label,
                stats_top_n,
                pad_frac,
            ],
            outputs=[
                records_state,
                idx_state,
                label_choices_state,
                summary,
                image,
                meta,
                review_label,
                note,
                corrected_label,
                progress,
                gallery,
                sample_table,
                label_bar,
                label_table,
            ],
        )
        prev_btn.click(
            on_prev,
            inputs=[records_state, idx_state, pad_frac],
            outputs=[idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        skip_btn.click(
            on_skip,
            inputs=[records_state, idx_state, pad_frac],
            outputs=[idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        save_next_btn.click(
            on_save_next,
            inputs=[records_state, idx_state, review_label, note, corrected_label, pad_frac],
            outputs=[records_state, idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        ok_btn.click(
            lambda records, idx, note_value, corrected_label_value, pad_frac_value: quick_save(
                records,
                idx,
                note_value,
                corrected_label_value,
                pad_frac_value,
                constants.NOISE_REVIEW_LABEL_OK,
            ),
            inputs=[records_state, idx_state, note, corrected_label, pad_frac],
            outputs=[records_state, idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        wrong_btn.click(
            lambda records, idx, note_value, corrected_label_value, pad_frac_value: quick_save(
                records,
                idx,
                note_value,
                corrected_label_value,
                pad_frac_value,
                constants.NOISE_REVIEW_LABEL_WRONG_LABEL,
            ),
            inputs=[records_state, idx_state, note, corrected_label, pad_frac],
            outputs=[records_state, idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        use_pred_btn.click(
            current_pred_label,
            inputs=[records_state, idx_state],
            outputs=[corrected_label],
        )
        wrong_pred_btn.click(
            wrong_with_pred_label,
            inputs=[records_state, idx_state, note, pad_frac],
            outputs=[records_state, idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        export_btn.click(export_sample_csv, inputs=[records_state], outputs=[export_file])

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 label 分布与抽样复核 Gradio UI。")
    parser.add_argument("--config", type=str, default=None, help="pipeline_config.yaml 路径。")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Gradio 服务 host。")
    parser.add_argument("--port", type=int, default=7862, help="Gradio 服务端口。")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器。")
    return parser.parse_args()


def main() -> None:
    global CONFIG, DB_PATH

    args = parse_args()
    CONFIG = utils.load_pipeline_config(args.config)
    utils.init_db(config=CONFIG)
    DB_PATH = Path(utils.join_data_root(CONFIG["path"]["db_path"], config=CONFIG))
    ensure_review_columns()

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
