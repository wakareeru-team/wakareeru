from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
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


REVIEW_CONFIG = {
    "score_source": "db",  # db | loss_round
    "loss_round": "latest",
    "score_col": "noise_score_v1",
    "skip_reviewed": True,
    "sampling_density": "quantile",  # quantile | linear | high_dense
    "bin_count": 5,
    "samples_per_bin": 8,
    "random_seed": 42,
    "pad_frac": 0.04,
    "review_labels": constants.NOISE_REVIEW_LABELS,
    "review_label_col": "noise_review_label",
    "review_note_col": "noise_review_note",
    "reviewed_at_col": "noise_reviewed_at",
    "review_score_col": "noise_review_score_col",
    "corrected_label_col": "manual_corrected_label",
    "corrected_at_col": "manual_corrected_at",
}

REVIEW_COLUMN_DEFS = {
    REVIEW_CONFIG["review_label_col"]: "TEXT",
    REVIEW_CONFIG["review_note_col"]: "TEXT",
    REVIEW_CONFIG["reviewed_at_col"]: "TEXT",
    REVIEW_CONFIG["review_score_col"]: "TEXT",
    REVIEW_CONFIG["corrected_label_col"]: "TEXT",
    REVIEW_CONFIG["corrected_at_col"]: "TEXT",
}

CONFIG: dict[str, Any] = {}
DB_PATH: Path
DATA_ROOT: Path


def quote_ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQLite identifier: {name!r}")
    return f'"{name}"'


def crop_columns(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("PRAGMA table_info(crops)").fetchall()
    return {row[1]: (row[2] or "") for row in rows}


def ensure_review_columns(db_path: Path | None = None) -> None:
    db_path = db_path or DB_PATH
    with sqlite3.connect(db_path) as conn:
        cols = crop_columns(conn)
        for col, col_type in REVIEW_COLUMN_DEFS.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE crops ADD COLUMN {quote_ident(col)} {col_type}")
        conn.commit()


def score_columns(db_path: Path | None = None) -> list[str]:
    db_path = db_path or DB_PATH
    with sqlite3.connect(db_path) as conn:
        cols = crop_columns(conn)
    preferred = [c for c in cols if c.startswith("noise_score")]
    numeric = [
        c for c, t in cols.items()
        if c not in preferred and any(token in t.upper() for token in ["REAL", "INT", "NUM"])
    ]
    return preferred + numeric


def label_expr_for_granularity(label_granularity: str) -> str:
    if label_granularity == "submodel":
        return "COALESCE(i.submodel, i.fine_grained_series, i.series)"
    if label_granularity == "fine_grained_series":
        return "COALESCE(i.fine_grained_series, i.series)"
    if label_granularity == "series":
        return "i.series"
    raise ValueError(
        "noise_detection.label_granularity must be one of: "
        "series, fine_grained_series, submodel"
    )


def known_label_choices(db_path: Path | None = None) -> list[str]:
    db_path = db_path or DB_PATH
    label_granularity = CONFIG["noise_detection"]["label_granularity"]
    label_expr = label_expr_for_granularity(label_granularity)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT {label_expr} AS label
            FROM images i
            WHERE {label_expr} IS NOT NULL
              AND TRIM({label_expr}) != ''
            ORDER BY label
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) and col != "crop_id"
    ]


def load_loss_round_features(loss_round: str) -> pd.DataFrame:
    round_dir = utils.get_loss_round_dir(config=CONFIG, active_round=loss_round)
    feature_path = round_dir / CONFIG["loss_analysis"]["loss_feature_file_name"]
    if not feature_path.exists():
        raise FileNotFoundError(f"Loss feature CSV not found: {feature_path}")

    features = pd.read_csv(feature_path)
    if (
        "noise_score_v1" not in features.columns
        and {"loss_mean_pct_in_label", "error_rate"}.issubset(features.columns)
    ):
        features["noise_score_v1"] = features["loss_mean_pct_in_label"] + features["error_rate"]
    return features


def loss_round_score_columns(loss_round: str) -> list[str]:
    features = load_loss_round_features(loss_round)
    preferred = [col for col in ["noise_score_v1", "loss_mean_pct_in_label", "error_rate", "mean", "loss_tail_mean"] if col in features.columns]
    others = [col for col in _numeric_columns(features) if col not in preferred]
    return preferred + others


def load_candidate_rows(
    score_col: str,
    skip_reviewed: bool = True,
    db_path: Path | None = None,
) -> pd.DataFrame:
    db_path = db_path or DB_PATH
    ensure_review_columns(db_path)
    if corrected_label and corrected_label not in set(known_label_choices(db_path)):
        raise gr.Error(f"Correct label 不在当前已知标签中: {corrected_label}")
    with sqlite3.connect(db_path) as conn:
        cols = crop_columns(conn)
        if score_col not in cols:
            raise ValueError(f"score_col={score_col!r} not found in crops table")

        score_expr = f"c.{quote_ident(score_col)}"
        review_label_col = quote_ident(REVIEW_CONFIG["review_label_col"])
        review_note_col = quote_ident(REVIEW_CONFIG["review_note_col"])
        reviewed_at_col = quote_ident(REVIEW_CONFIG["reviewed_at_col"])
        corrected_label_col = quote_ident(REVIEW_CONFIG["corrected_label_col"])
        where = [f"{score_expr} IS NOT NULL", "i.downloaded_path IS NOT NULL"]
        if skip_reviewed:
            where.append(f"(c.{review_label_col} IS NULL OR c.{review_label_col} = '')")

        sql = f"""
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
                {score_expr} AS score,
                c.{review_label_col} AS review_label,
                c.{review_note_col} AS review_note,
                c.{reviewed_at_col} AS reviewed_at,
                c.{corrected_label_col} AS corrected_label,
                i.file_title,
                i.downloaded_path,
                COALESCE(i.fine_grained_series, i.series) AS image_label,
                i.category
            FROM crops c
            JOIN images i ON i.id = c.image_id
            WHERE {' AND '.join(where)}
            ORDER BY score DESC
        """
        return pd.read_sql_query(sql, conn)


def load_candidate_rows_from_loss_round(
    loss_round: str,
    score_col: str,
    skip_reviewed: bool = True,
    db_path: Path | None = None,
) -> pd.DataFrame:
    db_path = db_path or DB_PATH
    ensure_review_columns(db_path)
    features = load_loss_round_features(loss_round)
    if "crop_id" not in features.columns:
        raise ValueError("loss feature CSV missing required column: crop_id")
    if score_col not in features.columns:
        raise ValueError(f"score_col={score_col!r} not found in loss feature CSV")

    feature_cols = []
    for col in [
        "crop_id",
        score_col,
        "label",
        "pred_label",
        "pred_label_rate",
        "mean",
        "loss_tail_mean",
        "error_rate",
        "loss_mean_pct_in_label",
    ]:
        if col in features.columns and col not in feature_cols:
            feature_cols.append(col)
    score_features = features[feature_cols].copy()
    score_features = score_features.rename(columns={score_col: "score"})
    score_features["crop_id"] = score_features["crop_id"].astype(int)

    with sqlite3.connect(db_path) as conn:
        review_label_col = quote_ident(REVIEW_CONFIG["review_label_col"])
        review_note_col = quote_ident(REVIEW_CONFIG["review_note_col"])
        reviewed_at_col = quote_ident(REVIEW_CONFIG["reviewed_at_col"])
        corrected_label_col = quote_ident(REVIEW_CONFIG["corrected_label_col"])
        where = ["i.downloaded_path IS NOT NULL"]
        sql = f"""
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
                c.{review_label_col} AS review_label,
                c.{review_note_col} AS review_note,
                c.{reviewed_at_col} AS reviewed_at,
                c.{corrected_label_col} AS corrected_label,
                i.file_title,
                i.downloaded_path,
                COALESCE(i.fine_grained_series, i.series) AS image_label,
                i.category
            FROM crops c
            JOIN images i ON i.id = c.image_id
            WHERE {' AND '.join(where)}
            ORDER BY c.id
        """
        metadata = pd.read_sql_query(sql, conn)

    candidates = metadata.merge(score_features, on="crop_id", how="inner")
    if skip_reviewed:
        candidates = candidates[
            candidates["review_label"].isna() | (candidates["review_label"].astype(str).str.strip() == "")
        ].copy()
    return candidates.sort_values("score", ascending=False).reset_index(drop=True)


def assign_score_bins(df: pd.DataFrame, bins: int, density: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="Int64")

    score = df["score"].astype(float)
    bins = max(1, int(bins))

    if density == "linear":
        return pd.cut(score, bins=bins, labels=False, duplicates="drop")

    rank_pct = score.rank(method="first", pct=True)
    if density == "high_dense":
        q = 1.0 - (1.0 - np.linspace(0, 1, bins + 1)) ** 2
        q[0], q[-1] = 0.0, 1.0
        edges = np.unique(q)
        return pd.cut(rank_pct, bins=edges, labels=False, include_lowest=True, duplicates="drop")

    if density == "quantile":
        return pd.qcut(score.rank(method="first"), q=bins, labels=False, duplicates="drop")

    raise ValueError("sampling_density must be one of: quantile, linear, high_dense")


def stratified_sample_rows(
    df: pd.DataFrame,
    bins: int,
    samples_per_bin: int,
    density: str,
    seed: int,
) -> pd.DataFrame:
    work = df.dropna(subset=["score"]).copy()
    if work.empty:
        return work.assign(score_bin=pd.Series(dtype="Int64"))

    work["score_bin"] = assign_score_bins(work, bins=bins, density=density)
    work = work.dropna(subset=["score_bin"]).copy()
    work["score_bin"] = work["score_bin"].astype(int)

    parts = []
    for _, group in work.groupby("score_bin", sort=False):
        n = min(int(samples_per_bin), len(group))
        parts.append(group.sample(n=n, random_state=int(seed)))
    if not parts:
        return work.head(0)

    sampled = pd.concat(parts, ignore_index=True)
    sampled = sampled.sort_values(["score_bin", "score"], ascending=[False, False]).reset_index(drop=True)
    return sampled


def load_review_sample(
    score_source: str,
    loss_round: str,
    score_col: str,
    skip_reviewed: bool,
    density: str,
    bins: int,
    samples_per_bin: int,
    seed: int,
) -> pd.DataFrame:
    if score_source == "loss_round":
        candidates = load_candidate_rows_from_loss_round(
            loss_round=loss_round,
            score_col=score_col,
            skip_reviewed=skip_reviewed,
        )
    else:
        candidates = load_candidate_rows(score_col=score_col, skip_reviewed=skip_reviewed)
    return stratified_sample_rows(
        candidates,
        bins=bins,
        samples_per_bin=samples_per_bin,
        density=density,
        seed=seed,
    )


def placeholder_image(message: str, size: tuple[int, int] = (512, 384)) -> Image.Image:
    img = Image.new("RGB", size, "#f2f2f2")
    draw = ImageDraw.Draw(img)
    draw.multiline_text((20, 20), message, fill="#333333")
    return img


def load_crop_image(row: dict[str, Any], pad_frac: float | None = None) -> Image.Image:
    pad = REVIEW_CONFIG["pad_frac"] if pad_frac is None else float(pad_frac)
    try:
        return utils.load_crop(row, config=CONFIG, pad_frac=pad)
    except Exception as exc:
        return placeholder_image(f"Failed to load crop_id={row.get('crop_id')}\n{exc}")


def row_markdown(row: dict[str, Any], idx: int, total: int) -> str:
    if not row:
        return "No sample loaded."

    fields = [
        ("progress", f"{idx + 1}/{total}"),
        ("crop_id", row.get("crop_id")),
        ("score", f"{float(row.get('score')):.6f}" if pd.notna(row.get("score")) else None),
        ("score_bin", row.get("score_bin")),
        ("series", row.get("series")),
        ("image_label", row.get("image_label")),
        ("corrected_label", row.get("corrected_label")),
        ("loss_label", row.get("label")),
        ("pred_label", row.get("pred_label")),
        ("pred_label_rate", f"{float(row.get('pred_label_rate')):.4f}" if pd.notna(row.get("pred_label_rate")) else None),
        ("error_rate", f"{float(row.get('error_rate')):.4f}" if pd.notna(row.get("error_rate")) else None),
        ("loss_tail_mean", f"{float(row.get('loss_tail_mean')):.4f}" if pd.notna(row.get("loss_tail_mean")) else None),
        ("power_type", row.get("power_type")),
        ("detector", row.get("detector_label")),
        ("detector_score", f"{float(row.get('detector_score')):.4f}" if pd.notna(row.get("detector_score")) else None),
        ("file_title", row.get("file_title")),
        ("category", row.get("category")),
        ("existing_review", row.get("review_label")),
        ("reviewed_at", row.get("reviewed_at")),
    ]
    lines = ["### Crop review"]
    for key, value in fields:
        if value is not None and value == value:
            lines.append(f"- **{key}**: {value}")
    return "\n".join(lines)


def display_record(records: list[dict[str, Any]], idx: int):
    total = len(records or [])
    if total == 0:
        return placeholder_image("No rows sampled."), "No rows sampled.", None, "", None, "0/0"

    idx = int(idx)
    if idx >= total:
        message = f"Review complete. {total}/{total} sampled crops have been visited."
        return placeholder_image(message), f"### Review complete\n\n{message}", None, "", None, f"{total}/{total} complete"

    idx = max(0, idx)
    row = records[idx]
    image = load_crop_image(row)
    md = row_markdown(row, idx, total)
    label = row.get("review_label") if row.get("review_label") else None
    note = row.get("review_note") or ""
    corrected_label = row.get("corrected_label") if row.get("corrected_label") else None
    progress = f"{idx + 1}/{total}"
    return image, md, label, note, corrected_label, progress


def save_review(
    crop_id: int,
    label: str,
    note: str,
    corrected_label: str | None,
    score_col: str,
    db_path: Path | None = None,
) -> None:
    if not label:
        raise gr.Error("请选择一个 review label 再保存。")
    corrected_label = str(corrected_label or "").strip() or None
    if label == constants.NOISE_REVIEW_LABEL_WRONG_LABEL and not corrected_label:
        raise gr.Error("标记为 wrong_label 时，请从 Correct label 里选择正确标签。")
    if label != constants.NOISE_REVIEW_LABEL_WRONG_LABEL:
        corrected_label = None

    db_path = db_path or DB_PATH
    ensure_review_columns(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            UPDATE crops
            SET {quote_ident(REVIEW_CONFIG['review_label_col'])} = ?,
                {quote_ident(REVIEW_CONFIG['review_note_col'])} = ?,
                {quote_ident(REVIEW_CONFIG['reviewed_at_col'])} = CURRENT_TIMESTAMP,
                {quote_ident(REVIEW_CONFIG['review_score_col'])} = ?,
                {quote_ident(REVIEW_CONFIG['corrected_label_col'])} = ?,
                {quote_ident(REVIEW_CONFIG['corrected_at_col'])} =
                    CASE WHEN ? IS NULL THEN NULL ELSE CURRENT_TIMESTAMP END
            WHERE id = ?
            """,
            (label, note or None, score_col, corrected_label, corrected_label, int(crop_id)),
        )
        conn.commit()


def build_review_app():
    score_choices = score_columns()
    label_choices = known_label_choices()
    default_score_col = REVIEW_CONFIG["score_col"] if REVIEW_CONFIG["score_col"] in score_choices else (
        score_choices[0] if score_choices else None
    )

    with gr.Blocks(title="Noise Crop Review") as app:
        gr.Markdown("# Noise Crop Review")

        records_state = gr.State([])
        idx_state = gr.State(0)

        with gr.Row():
            with gr.Column(scale=1):
                score_col = gr.Dropdown(
                    choices=score_choices,
                    value=default_score_col,
                    label="Score column",
                    allow_custom_value=True,
                )
                score_source = gr.Radio(
                    choices=["db", "loss_round"],
                    value=REVIEW_CONFIG["score_source"],
                    label="Score source",
                )
                loss_round = gr.Textbox(
                    value=REVIEW_CONFIG["loss_round"],
                    label="Loss round",
                    placeholder="latest or 20260528_214343",
                )
                refresh_score_cols_btn = gr.Button("Refresh score columns")
                skip_reviewed = gr.Checkbox(
                    value=REVIEW_CONFIG["skip_reviewed"],
                    label="Skip already reviewed crops",
                )
                density = gr.Radio(
                    choices=["high_dense", "quantile", "linear"],
                    value=REVIEW_CONFIG["sampling_density"],
                    label="Sampling score density",
                )
                bins = gr.Slider(2, 20, value=REVIEW_CONFIG["bin_count"], step=1, label="Bin count")
                samples_per_bin = gr.Slider(
                    1,
                    50,
                    value=REVIEW_CONFIG["samples_per_bin"],
                    step=1,
                    label="Samples per bin",
                )
                seed = gr.Number(value=REVIEW_CONFIG["random_seed"], precision=0, label="Random seed")
                reload_btn = gr.Button("Load / resample", variant="primary")
                progress = gr.Textbox(label="Progress", interactive=False)

            with gr.Column(scale=2):
                image = gr.Image(label="Crop", type="pil", height=520)
                meta = gr.Markdown()

            with gr.Column(scale=1):
                review_label = gr.Radio(
                    choices=REVIEW_CONFIG["review_labels"],
                    label="Manual review label",
                )
                with gr.Row():
                    quick_ok_btn = gr.Button("OK", variant="secondary")
                    quick_wrong_label_btn = gr.Button("Wrong Label", variant="secondary")
                with gr.Row():
                    quick_out_of_label_space_btn = gr.Button("Out of Label Space", variant="secondary")
                    quick_bad_crop_btn = gr.Button("Bad Crop", variant="stop")
                with gr.Row():
                    quick_ambiguous_btn = gr.Button("Ambiguous", variant="secondary")
                note = gr.Textbox(label="Note", lines=4, placeholder="optional")
                corrected_label = gr.Dropdown(
                    choices=label_choices,
                    label="Correct label",
                    allow_custom_value=False,
                )
                refresh_labels_btn = gr.Button("Refresh labels")
                with gr.Row():
                    prev_btn = gr.Button("Previous")
                    skip_btn = gr.Button("Skip")
                save_next_btn = gr.Button("Save & next", variant="primary")

        sample_table = gr.Dataframe(label="Current sampled rows", interactive=False, wrap=True)

        def on_refresh_score_columns(score_source_value, loss_round_value):
            if score_source_value == "loss_round":
                choices = loss_round_score_columns(str(loss_round_value).strip() or "latest")
            else:
                choices = score_columns()
            value = REVIEW_CONFIG["score_col"] if REVIEW_CONFIG["score_col"] in choices else (
                choices[0] if choices else None
            )
            return gr.update(choices=choices, value=value)

        def on_reload(score_source_value, loss_round_value, score_col_value, skip_value, density_value, bins_value, samples_per_bin_value, seed_value):
            sample = load_review_sample(
                score_source=score_source_value,
                loss_round=str(loss_round_value).strip() or "latest",
                score_col=score_col_value,
                skip_reviewed=bool(skip_value),
                density=density_value,
                bins=int(bins_value),
                samples_per_bin=int(samples_per_bin_value),
                seed=int(seed_value),
            )
            records = sample.to_dict(orient="records")
            img, md, label, note_value, corrected_label_value, prog = display_record(records, 0)
            preview_cols = [
                c for c in [
                    "crop_id", "score", "score_bin", "series", "image_label",
                    "corrected_label",
                    "label", "pred_label", "pred_label_rate", "error_rate",
                    "detector_label", "file_title", "review_label", "reviewed_at",
                ]
                if c in sample.columns
            ]
            return records, 0, img, md, label, note_value, corrected_label_value, prog, sample[preview_cols]

        def score_col_for_review(score_source_value, loss_round_value, score_col_value):
            if score_source_value == "loss_round":
                return f"loss_round:{str(loss_round_value).strip() or 'latest'}:{score_col_value}"
            return str(score_col_value)

        def on_save_next(records, idx, label, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value):
            records = records or []
            if not records:
                raise gr.Error("当前没有 sample，请先 Load / resample。")
            idx = int(idx)
            if idx >= len(records):
                raise gr.Error("这一批样本已经全部 review 完成。")
            idx = max(0, idx)
            row = records[idx]
            save_review(
                row["crop_id"],
                label,
                note_value,
                corrected_label_value,
                score_col_for_review(score_source_value, loss_round_value, score_col_value),
            )
            row["review_label"] = label
            row["review_note"] = note_value
            row["corrected_label"] = corrected_label_value if label == constants.NOISE_REVIEW_LABEL_WRONG_LABEL else None
            idx = idx + 1
            img, md, next_label, next_note, next_corrected_label, prog = display_record(records, idx)
            return records, idx, img, md, next_label, next_note, next_corrected_label, prog

        def on_quick_save_next(records, idx, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value, label):
            return on_save_next(records, idx, label, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value)

        def on_skip(records, idx):
            records = records or []
            if not records:
                return 0, *display_record(records, 0)
            idx = min(int(idx) + 1, len(records))
            img, md, label, note_value, corrected_label_value, prog = display_record(records, idx)
            return idx, img, md, label, note_value, corrected_label_value, prog

        def on_prev(records, idx):
            records = records or []
            if not records:
                return 0, *display_record(records, 0)
            idx = max(0, int(idx) - 1)
            img, md, label, note_value, corrected_label_value, prog = display_record(records, idx)
            return idx, img, md, label, note_value, corrected_label_value, prog

        def on_refresh_labels():
            choices = known_label_choices()
            return gr.update(choices=choices, value=None)

        refresh_score_cols_btn.click(
            on_refresh_score_columns,
            inputs=[score_source, loss_round],
            outputs=[score_col],
        )
        score_source.change(
            on_refresh_score_columns,
            inputs=[score_source, loss_round],
            outputs=[score_col],
        )
        reload_btn.click(
            on_reload,
            inputs=[score_source, loss_round, score_col, skip_reviewed, density, bins, samples_per_bin, seed],
            outputs=[records_state, idx_state, image, meta, review_label, note, corrected_label, progress, sample_table],
        )
        save_next_btn.click(
            on_save_next,
            inputs=[records_state, idx_state, review_label, note, corrected_label, score_source, loss_round, score_col],
            outputs=[records_state, idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        quick_ok_btn.click(
            lambda records, idx, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value: on_quick_save_next(
                records, idx, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value, constants.NOISE_REVIEW_LABEL_OK
            ),
            inputs=[records_state, idx_state, note, corrected_label, score_source, loss_round, score_col],
            outputs=[records_state, idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        quick_wrong_label_btn.click(
            lambda records, idx, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value: on_quick_save_next(
                records, idx, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value, constants.NOISE_REVIEW_LABEL_WRONG_LABEL
            ),
            inputs=[records_state, idx_state, note, corrected_label, score_source, loss_round, score_col],
            outputs=[records_state, idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        quick_out_of_label_space_btn.click(
            lambda records, idx, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value: on_quick_save_next(
                records, idx, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value, constants.NOISE_REVIEW_LABEL_OUT_OF_LABEL_SPACE
            ),
            inputs=[records_state, idx_state, note, corrected_label, score_source, loss_round, score_col],
            outputs=[records_state, idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        quick_bad_crop_btn.click(
            lambda records, idx, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value: on_quick_save_next(
                records, idx, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value, constants.NOISE_REVIEW_LABEL_BAD_CROP
            ),
            inputs=[records_state, idx_state, note, corrected_label, score_source, loss_round, score_col],
            outputs=[records_state, idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        quick_ambiguous_btn.click(
            lambda records, idx, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value: on_quick_save_next(
                records, idx, note_value, corrected_label_value, score_source_value, loss_round_value, score_col_value, constants.NOISE_REVIEW_LABEL_AMBIGUOUS
            ),
            inputs=[records_state, idx_state, note, corrected_label, score_source, loss_round, score_col],
            outputs=[records_state, idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        refresh_labels_btn.click(
            on_refresh_labels,
            inputs=[],
            outputs=[corrected_label],
        )
        skip_btn.click(
            on_skip,
            inputs=[records_state, idx_state],
            outputs=[idx_state, image, meta, review_label, note, corrected_label, progress],
        )
        prev_btn.click(
            on_prev,
            inputs=[records_state, idx_state],
            outputs=[idx_state, image, meta, review_label, note, corrected_label, progress],
        )

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Wakareeru noise review Gradio UI.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline_config.yaml.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Gradio server host.")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    return parser.parse_args()


def main() -> None:
    global CONFIG, DB_PATH, DATA_ROOT

    args = parse_args()
    CONFIG = utils.load_pipeline_config(args.config)
    utils.init_db(config=CONFIG)
    DB_PATH = Path(utils.join_data_root(CONFIG["path"]["db_path"], config=CONFIG))
    DATA_ROOT = utils.get_data_root(CONFIG)
    ensure_review_columns(DB_PATH)

    app = build_review_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
