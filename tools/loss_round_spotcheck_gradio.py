from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import tempfile
import time
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

from pipeline import utils  # noqa: E402


SPOTCHECK_CONFIG = {
    "loss_round": "latest",
    "sample_mode": "lr_high_score",
    "score_col": "noise_predicted_prob",
    "sample_size": 48,
    "samples_per_label": 4,
    "random_seed": 42,
    "gallery_limit": 80,
    "crop_pad_frac": 0.04,
}

CONFIG: dict[str, Any] = {}
DB_PATH: Path


def quote_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQLite identifier: {name!r}")
    return f'"{name}"'


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({quote_ident(table_name)})")}


def resolve_loss_round(loss_round: str) -> tuple[str, Path]:
    active_round = str(loss_round).strip() or "latest"
    round_dir = utils.get_loss_round_dir(config=CONFIG, active_round=active_round)
    return round_dir.name, round_dir


def load_loss_features(loss_round: str) -> pd.DataFrame:
    _, round_dir = resolve_loss_round(loss_round)
    feature_path = round_dir / CONFIG["loss_analysis"]["loss_feature_file_name"]
    if not feature_path.exists():
        raise FileNotFoundError(f"Loss feature CSV not found: {feature_path}")

    features = pd.read_csv(feature_path)
    if "crop_id" not in features.columns:
        raise ValueError(f"Loss feature CSV missing required column: {feature_path}")
    if (
        "noise_score_v1" not in features.columns
        and {"loss_mean_pct_in_label", "error_rate"}.issubset(features.columns)
    ):
        features["noise_score_v1"] = features["loss_mean_pct_in_label"] + features["error_rate"]
    features["crop_id"] = features["crop_id"].astype(int)
    return features


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
        "noise_predicted_prob",
        "noise_predicted_label",
        "noise_prediction_model",
        "predicted_at",
    ]
    predictions = predictions[[col for col in keep_cols if col in predictions.columns]].copy()
    predictions["crop_id"] = predictions["crop_id"].astype(int)
    return predictions


def score_columns(loss_round: str) -> list[str]:
    rows = load_round_rows(loss_round)
    preferred = [
        "noise_predicted_prob",
        "noise_score_v1",
        "error_rate",
        "loss_mean_pct_in_label",
        "mean",
        "loss_tail_mean",
        "pred_label_rate",
    ]
    numeric = [
        col
        for col in rows.columns
        if col != "crop_id" and pd.api.types.is_numeric_dtype(rows[col])
    ]
    return [col for col in preferred if col in numeric] + [
        col for col in numeric if col not in preferred
    ]


def metadata_sql() -> str:
    optional_crop_cols = {
        "noise_review_label": "c.noise_review_label AS noise_review_label",
        "manual_corrected_label": "c.manual_corrected_label AS manual_corrected_label",
        "noise_predicted_label": "c.noise_predicted_label AS db_noise_predicted_label",
        "noise_predicted_prob": "c.noise_predicted_prob AS db_noise_predicted_prob",
        "noise_prediction_model": "c.noise_prediction_model AS db_noise_prediction_model",
    }
    optional_image_cols = {
        "submodel": "i.submodel AS submodel",
        "fine_grained_series": "i.fine_grained_series AS fine_grained_series",
        "bandai": "i.bandai AS bandai",
        "operator_jp": "i.operator_jp AS operator_jp",
        "operator_en": "i.operator_en AS operator_en",
    }

    with sqlite3.connect(DB_PATH) as conn:
        crop_cols = table_columns(conn, "crops")
        image_cols = table_columns(conn, "images")

    optional_selects = [
        expr for col, expr in optional_crop_cols.items() if col in crop_cols
    ] + [
        expr for col, expr in optional_image_cols.items() if col in image_cols
    ]
    optional_sql = ",\n                " + ",\n                ".join(optional_selects) if optional_selects else ""

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
            i.category{optional_sql}
        FROM crops c
        JOIN images i ON i.id = c.image_id
        WHERE i.downloaded_path IS NOT NULL
          AND TRIM(i.downloaded_path) != ''
        ORDER BY c.id
    """


def load_round_rows(loss_round: str) -> pd.DataFrame:
    features = load_loss_features(loss_round)
    lr_predictions = load_lr_prediction_file(loss_round)
    with sqlite3.connect(DB_PATH) as conn:
        metadata = pd.read_sql_query(metadata_sql(), conn)
    rows = metadata.merge(features, on="crop_id", how="inner")
    if not lr_predictions.empty:
        rows = rows.merge(lr_predictions, on="crop_id", how="left")
    if "noise_predicted_prob" not in rows.columns and "db_noise_predicted_prob" in rows.columns:
        rows["noise_predicted_prob"] = rows["db_noise_predicted_prob"]
    elif "db_noise_predicted_prob" in rows.columns:
        rows["noise_predicted_prob"] = rows["noise_predicted_prob"].fillna(
            rows["db_noise_predicted_prob"]
        )
    if "noise_predicted_label" not in rows.columns and "db_noise_predicted_label" in rows.columns:
        rows["noise_predicted_label"] = rows["db_noise_predicted_label"]
    elif "db_noise_predicted_label" in rows.columns:
        rows["noise_predicted_label"] = rows["noise_predicted_label"].fillna(
            rows["db_noise_predicted_label"]
        )
    if "noise_prediction_model" not in rows.columns and "db_noise_prediction_model" in rows.columns:
        rows["noise_prediction_model"] = rows["db_noise_prediction_model"]
    elif "db_noise_prediction_model" in rows.columns:
        rows["noise_prediction_model"] = rows["noise_prediction_model"].fillna(
            rows["db_noise_prediction_model"]
        )
    if "label" not in rows.columns:
        rows["label"] = rows.get("fine_grained_series", rows["series"])
    rows["label"] = rows["label"].fillna(rows["series"]).astype(str)
    return rows


def filter_rows(rows: pd.DataFrame, label_query: str, only_mismatch: bool, skip_reviewed: bool) -> pd.DataFrame:
    filtered = rows.copy()
    label_query = str(label_query or "").strip()
    if label_query:
        mask = filtered["label"].astype(str).str.contains(label_query, case=False, regex=False, na=False)
        series_mask = filtered["series"].astype(str).str.contains(label_query, case=False, regex=False, na=False)
        filtered = filtered[mask | series_mask].copy()
    if only_mismatch and {"label", "pred_label"}.issubset(filtered.columns):
        filtered = filtered[
            filtered["pred_label"].notna()
            & (filtered["label"].astype(str) != filtered["pred_label"].astype(str))
        ].copy()
    if skip_reviewed and "noise_review_label" in filtered.columns:
        reviewed = filtered["noise_review_label"].fillna("").astype(str).str.strip() != ""
        filtered = filtered[~reviewed].copy()
    return filtered


def sample_rows(
    rows: pd.DataFrame,
    sample_mode: str,
    score_col: str,
    sample_size: int,
    samples_per_label: int,
    seed: int,
) -> pd.DataFrame:
    if rows.empty:
        return rows

    sample_size = max(1, int(sample_size))
    samples_per_label = max(1, int(samples_per_label))
    seed = int(seed)
    work = rows.copy()

    if sample_mode == "lr_high_score":
        if "noise_predicted_prob" not in work.columns:
            raise gr.Error("当前轮次没有 LR 分数。请先运行 stage 13 生成 noise_predicted_prob。")
        if not work["noise_predicted_prob"].notna().any():
            raise gr.Error("当前轮次的 LR 分数为空。请先运行 stage 13 生成 noise_predicted_prob。")
        return (
            work.dropna(subset=["noise_predicted_prob"])
            .sort_values("noise_predicted_prob", ascending=False)
            .head(sample_size)
            .reset_index(drop=True)
        )

    if sample_mode == "lr_predicted_noise":
        if "noise_predicted_label" not in work.columns:
            raise gr.Error("当前轮次没有 LR 预测标签。请先运行 stage 13 生成 noise_predicted_label。")
        if not work["noise_predicted_label"].notna().any():
            raise gr.Error("当前轮次的 LR 预测标签为空。请先运行 stage 13。")
        positive_labels = set(CONFIG["logistic_regression_filter"]["noise_positive_label"])
        work = work[work["noise_predicted_label"].isin(positive_labels)].copy()
        if "noise_predicted_prob" in work.columns:
            work = work.sort_values("noise_predicted_prob", ascending=False)
        return work.head(sample_size).reset_index(drop=True)

    if sample_mode == "lr_uncertain":
        if "noise_predicted_prob" not in work.columns:
            raise gr.Error("当前轮次没有 LR 分数。请先运行 stage 13 生成 noise_predicted_prob。")
        if not work["noise_predicted_prob"].notna().any():
            raise gr.Error("当前轮次的 LR 分数为空。请先运行 stage 13。")
        work = work.dropna(subset=["noise_predicted_prob"]).copy()
        work["lr_margin_from_0_5"] = (work["noise_predicted_prob"].astype(float) - 0.5).abs()
        return work.sort_values("lr_margin_from_0_5").head(sample_size).reset_index(drop=True)

    if sample_mode == "high_error_rate":
        if "error_rate" not in work.columns:
            raise gr.Error("当前轮次的 loss feature 中没有 error_rate 列。")
        return work.sort_values("error_rate", ascending=False).head(sample_size).reset_index(drop=True)

    if sample_mode == "label_balanced_random":
        return (
            work.groupby("label", group_keys=False)
            .apply(lambda group: group.sample(n=min(samples_per_label, len(group)), random_state=seed))
            .reset_index(drop=True)
        )

    if sample_mode == "label_balanced_high_suspicion":
        if score_col not in work.columns:
            raise gr.Error(f"当前轮次的 loss feature 中没有 {score_col!r} 列。")
        work = work.dropna(subset=[score_col]).copy()
        return (
            work.sort_values(score_col, ascending=False)
            .groupby("label", group_keys=False)
            .head(samples_per_label)
            .reset_index(drop=True)
        )

    if sample_mode == "prediction_mismatch":
        if "pred_label" not in work.columns:
            raise gr.Error("当前轮次的 loss feature 中没有 pred_label 列。")
        work = work[work["pred_label"].notna() & (work["label"].astype(str) != work["pred_label"].astype(str))]
        if score_col in work.columns:
            work = work.sort_values(score_col, ascending=False)
        return work.head(sample_size).reset_index(drop=True)

    if sample_mode == "random":
        return work.sample(n=min(sample_size, len(work)), random_state=seed).reset_index(drop=True)

    if score_col not in work.columns:
        raise gr.Error(f"当前轮次的 loss feature 中没有 {score_col!r} 列。")
    return work.dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(sample_size).reset_index(drop=True)


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


def caption_for_row(row: dict[str, Any], score_col: str) -> str:
    parts = [f"crop_id={row.get('crop_id')}", str(row.get("label"))]
    if "noise_predicted_prob" in row and pd.notna(row.get("noise_predicted_prob")):
        parts.append(f"LR={float(row['noise_predicted_prob']):.3f}")
    lr_label = row.get("noise_predicted_label")
    if lr_label is not None and lr_label == lr_label:
        parts.append(f"LR标签={lr_label}")
    if score_col != "noise_predicted_prob" and score_col in row and pd.notna(row.get(score_col)):
        parts.append(f"{score_col}={float(row[score_col]):.3f}")
    if "error_rate" in row and pd.notna(row.get("error_rate")):
        parts.append(f"错误率={float(row['error_rate']):.2f}")
    pred_label = row.get("pred_label")
    if pred_label is not None and pred_label == pred_label:
        parts.append(f"线性头预测={pred_label}")
    return " | ".join(parts)


def make_gallery(sample: pd.DataFrame, score_col: str, gallery_limit: int, pad_frac: float) -> list[tuple[Image.Image, str]]:
    items = []
    for row in sample.head(max(1, int(gallery_limit))).to_dict(orient="records"):
        items.append((load_crop_image(row, pad_frac), caption_for_row(row, score_col)))
    return items


def preview_columns(sample: pd.DataFrame, score_col: str) -> pd.DataFrame:
    columns = [
        "crop_id",
        "label",
        "series",
        "pred_label",
        "pred_label_rate",
        "noise_predicted_prob",
        "noise_predicted_label",
        "noise_prediction_model",
        score_col,
        "lr_margin_from_0_5",
        "error_rate",
        "loss_mean_pct_in_label",
        "mean",
        "loss_tail_mean",
        "detector_score",
        "crop_status",
        "noise_review_label",
        "manual_corrected_label",
        "file_title",
        "category",
        "downloaded_path",
    ]
    selected = list(dict.fromkeys(col for col in columns if col in sample.columns))
    return sample[selected].copy()


def summary_markdown(round_name: str, rows: pd.DataFrame, filtered: pd.DataFrame, sample: pd.DataFrame) -> str:
    label_count = sample["label"].nunique() if "label" in sample.columns and not sample.empty else 0
    lr_count = (
        int(sample["noise_predicted_prob"].notna().sum())
        if "noise_predicted_prob" in sample.columns and not sample.empty
        else 0
    )
    lines = [
        "### 抽样概览",
        f"- **loss round**: {round_name}",
        f"- **本轮样本数**: {len(rows)}",
        f"- **筛选后样本数**: {len(filtered)}",
        f"- **抽样数**: {len(sample)}",
        f"- **覆盖标签数**: {label_count}",
        f"- **含 LR 分数样本数**: {lr_count}",
    ]
    return "\n".join(lines)


def export_sample_csv(records: list[dict[str, Any]], loss_round: str) -> str | None:
    if not records:
        raise gr.Error("当前没有抽样结果可导出。")
    round_name, _ = resolve_loss_round(loss_round)
    output_dir = Path(tempfile.gettempdir()) / "wakareeru_spotcheck"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"spotcheck_{round_name}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return str(path)


def build_app() -> gr.Blocks:
    default_scores = score_columns(SPOTCHECK_CONFIG["loss_round"])
    default_score = SPOTCHECK_CONFIG["score_col"] if SPOTCHECK_CONFIG["score_col"] in default_scores else (
        default_scores[0] if default_scores else None
    )

    with gr.Blocks(title="Loss Round 抽查") as app:
        gr.Markdown("# Loss Round 抽查")
        records_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                loss_round = gr.Textbox(
                    value=SPOTCHECK_CONFIG["loss_round"],
                    label="Loss round",
                    placeholder="latest or 20260528_214343",
                )
                score_col = gr.Dropdown(
                    choices=default_scores,
                    value=default_score,
                    label="可疑分数列",
                    allow_custom_value=True,
                )
                refresh_scores = gr.Button("刷新分数列")
                sample_mode = gr.Radio(
                    choices=[
                        ("LR 高分样本", "lr_high_score"),
                        ("LR 预测为噪声", "lr_predicted_noise"),
                        ("LR 0.5 附近样本", "lr_uncertain"),
                        ("Loss 高可疑", "high_suspicion"),
                        ("高错误率", "high_error_rate"),
                        ("按标签均衡随机", "label_balanced_random"),
                        ("按标签均衡高可疑", "label_balanced_high_suspicion"),
                        ("线性头预测不一致", "prediction_mismatch"),
                        ("随机抽样", "random"),
                    ],
                    value=SPOTCHECK_CONFIG["sample_mode"],
                    label="抽样方式",
                )
                label_query = gr.Textbox(
                    value="",
                    label="标签 / series 包含",
                    placeholder="可选，例如 E231",
                )
                only_mismatch = gr.Checkbox(value=False, label="只看 label != 线性头预测")
                skip_reviewed = gr.Checkbox(value=False, label="跳过已人工复核 crop")
                sample_size = gr.Slider(1, 300, value=SPOTCHECK_CONFIG["sample_size"], step=1, label="抽样数量")
                samples_per_label = gr.Slider(
                    1,
                    30,
                    value=SPOTCHECK_CONFIG["samples_per_label"],
                    step=1,
                    label="每个标签抽样数",
                )
                seed = gr.Number(value=SPOTCHECK_CONFIG["random_seed"], precision=0, label="随机种子")
                gallery_limit = gr.Slider(
                    1,
                    160,
                    value=SPOTCHECK_CONFIG["gallery_limit"],
                    step=1,
                    label="图库显示上限",
                )
                pad_frac = gr.Slider(0.0, 0.25, value=SPOTCHECK_CONFIG["crop_pad_frac"], step=0.01, label="Crop 外扩比例")
                load_btn = gr.Button("加载抽样", variant="primary")
                export_btn = gr.Button("导出当前抽样 CSV")
                export_file = gr.File(label="导出的 CSV")

            with gr.Column(scale=3):
                summary = gr.Markdown()
                gallery = gr.Gallery(label="抽样 crop", columns=4, height=640, object_fit="contain")
                table = gr.Dataframe(label="抽样明细", interactive=False, wrap=True)

        def on_refresh(loss_round_value):
            choices = score_columns(str(loss_round_value).strip() or "latest")
            value = SPOTCHECK_CONFIG["score_col"] if SPOTCHECK_CONFIG["score_col"] in choices else (
                choices[0] if choices else None
            )
            return gr.update(choices=choices, value=value)

        def on_load(
            loss_round_value,
            score_col_value,
            sample_mode_value,
            label_query_value,
            only_mismatch_value,
            skip_reviewed_value,
            sample_size_value,
            samples_per_label_value,
            seed_value,
            gallery_limit_value,
            pad_frac_value,
        ):
            round_name, _ = resolve_loss_round(str(loss_round_value).strip() or "latest")
            rows = load_round_rows(str(loss_round_value).strip() or "latest")
            filtered = filter_rows(
                rows,
                label_query=str(label_query_value or ""),
                only_mismatch=bool(only_mismatch_value),
                skip_reviewed=bool(skip_reviewed_value),
            )
            sample = sample_rows(
                filtered,
                sample_mode=str(sample_mode_value),
                score_col=str(score_col_value),
                sample_size=int(sample_size_value),
                samples_per_label=int(samples_per_label_value),
                seed=int(seed_value),
            )
            records = sample.to_dict(orient="records")
            return (
                records,
                summary_markdown(round_name, rows, filtered, sample),
                make_gallery(sample, str(score_col_value), int(gallery_limit_value), float(pad_frac_value)),
                preview_columns(sample, str(score_col_value)),
            )

        refresh_scores.click(on_refresh, inputs=[loss_round], outputs=[score_col])
        loss_round.change(on_refresh, inputs=[loss_round], outputs=[score_col])
        load_btn.click(
            on_load,
            inputs=[
                loss_round,
                score_col,
                sample_mode,
                label_query,
                only_mismatch,
                skip_reviewed,
                sample_size,
                samples_per_label,
                seed,
                gallery_limit,
                pad_frac,
            ],
            outputs=[records_state, summary, gallery, table],
        )
        export_btn.click(export_sample_csv, inputs=[records_state, loss_round], outputs=[export_file])

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动只读 loss round 抽查 Gradio UI。")
    parser.add_argument("--config", type=str, default=None, help="pipeline_config.yaml 路径。")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Gradio 服务 host。")
    parser.add_argument("--port", type=int, default=7861, help="Gradio 服务端口。")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器。")
    return parser.parse_args()


def main() -> None:
    global CONFIG, DB_PATH

    args = parse_args()
    CONFIG = utils.load_pipeline_config(args.config)
    utils.init_db(config=CONFIG)
    DB_PATH = Path(utils.join_data_root(CONFIG["path"]["db_path"], config=CONFIG))

    app = build_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
