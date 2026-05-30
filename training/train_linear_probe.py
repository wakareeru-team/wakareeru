from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm.auto import tqdm


def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "pipeline").exists():
            return candidate
    raise RuntimeError(f"Cannot find wakareeru project root from {start}")


PROJECT_ROOT = find_project_root()
PIPELINE_DIR = PROJECT_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import utils  # noqa: E402


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, crop_ids: torch.Tensor):
        self.features = features.float()
        self.labels = labels.long()
        self.crop_ids = crop_ids.long()
        if not (len(self.features) == len(self.labels) == len(self.crop_ids)):
            raise ValueError("features, labels and crop_ids must have the same length")

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx], self.crop_ids[idx]


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a linear probe on the stored Wakareeru dataset.")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline_config.yaml")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run directory name")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override configured max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configured train batch size")
    parser.add_argument("--feature-cache-file", type=str, default=None, help="Override feature cache file")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_feature_cache_path(config: dict[str, Any], train_config: dict[str, Any]) -> Path:
    feature_cache_dir = utils.join_data_root(train_config["feature_cache_dir"], config=config)
    feature_cache_file = train_config["feature_cache_file"]
    if feature_cache_file == "latest":
        pointer_path = feature_cache_dir / train_config["latest_feature_cache_file"]
        if not pointer_path.exists():
            raise FileNotFoundError(f"Feature cache latest pointer not found: {pointer_path}")
        feature_cache_file = pointer_path.read_text(encoding="utf-8").strip()
        if not feature_cache_file:
            raise ValueError(f"Feature cache latest pointer is empty: {pointer_path}")
    feature_cache_path = Path(feature_cache_file)
    if not feature_cache_path.is_absolute():
        feature_cache_path = feature_cache_dir / feature_cache_path
    return feature_cache_path


def extract_crop_id(image_path: str) -> int:
    match = re.search(r"_(\d+)\.[^.]+$", image_path)
    if not match:
        raise ValueError(f"Cannot parse crop_id from image_path: {image_path}")
    return int(match.group(1))


def load_dataset_metadata(config: dict[str, Any], train_config: dict[str, Any]) -> pd.DataFrame:
    dataset_root = utils.join_data_root(train_config["dataset_dir"], config=config)
    metadata_path = dataset_root / train_config["metadata_file_name"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    label_column = train_config["label_column"]
    required_columns = {"image_path", label_column}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Dataset metadata missing columns: {sorted(missing_columns)}")

    metadata = metadata.copy()
    metadata[label_column] = metadata[label_column].astype("string").str.strip()
    metadata = metadata[metadata[label_column].notna() & (metadata[label_column] != "")].copy()
    metadata["crop_id"] = metadata["image_path"].astype(str).map(extract_crop_id)
    metadata["abs_image_path"] = metadata["image_path"].map(lambda path: dataset_root / str(path))
    if train_config["validate_image_files"]:
        exists = metadata["abs_image_path"].map(Path.exists)
        missing_count = int((~exists).sum())
        if missing_count:
            raise FileNotFoundError(f"Dataset image files missing: {missing_count}")
    return metadata.reset_index(drop=True)


def load_feature_cache(feature_cache_path: Path) -> dict[str, Any]:
    if not feature_cache_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {feature_cache_path}")
    cache = torch.load(feature_cache_path, map_location="cpu")
    required_keys = {"features", "crop_ids"}
    missing_keys = required_keys - set(cache)
    if missing_keys:
        raise KeyError(f"Feature cache missing keys: {sorted(missing_keys)}")
    if len(cache["features"]) != len(cache["crop_ids"]):
        raise ValueError("Feature cache features and crop_ids have different lengths")
    return cache


def build_training_frame(
    metadata: pd.DataFrame,
    feature_cache: dict[str, Any],
    train_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int], dict[int, str], dict[str, int]]:
    label_column = train_config["label_column"]
    crop_ids = feature_cache["crop_ids"].cpu().numpy().astype(int)
    feature_lookup = pd.DataFrame({"crop_id": crop_ids, "feature_index": np.arange(len(crop_ids))})
    duplicated_features = int(feature_lookup["crop_id"].duplicated().sum())
    if duplicated_features:
        raise ValueError(f"Feature cache contains duplicated crop_id rows: {duplicated_features}")

    before_feature_join = len(metadata)
    frame = metadata.merge(feature_lookup, on="crop_id", how="inner")
    missing_features = before_feature_join - len(frame)

    label_counts = frame[label_column].value_counts()
    min_samples = int(train_config["min_samples_per_class"])
    kept_labels = label_counts[label_counts >= min_samples].index
    excluded_by_count = int((~frame[label_column].isin(kept_labels)).sum())
    frame = frame[frame[label_column].isin(kept_labels)].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("No training samples remain after feature join and class-count filtering")

    labels = sorted(frame[label_column].unique())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    frame["train_label_id"] = frame[label_column].map(label_to_id).astype(int)

    report = {
        "metadata_rows": int(before_feature_join),
        "feature_cache_rows": int(len(crop_ids)),
        "missing_feature_rows": int(missing_features),
        "excluded_by_min_samples": int(excluded_by_count),
        "trainable_rows": int(len(frame)),
        "num_classes": int(len(label_to_id)),
    }
    return frame, label_to_id, id_to_label, report


def split_frame(frame: pd.DataFrame, train_config: dict[str, Any]) -> pd.DataFrame:
    seed = int(train_config["seed"])
    val_ratio = float(train_config["val_ratio"])
    test_ratio = float(train_config["test_ratio"])
    if not (0.0 < val_ratio < 1.0 and 0.0 < test_ratio < 1.0 and val_ratio + test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1")

    train_val, test = train_test_split(
        frame,
        test_size=test_ratio,
        random_state=seed,
        stratify=frame["train_label_id"],
    )
    relative_val_ratio = val_ratio / (1.0 - test_ratio)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_ratio,
        random_state=seed,
        stratify=train_val["train_label_id"],
    )
    split_parts = [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")]
    return pd.concat(split_parts, ignore_index=True).reset_index(drop=True)


def make_split_dataset(
    frame: pd.DataFrame,
    feature_cache: dict[str, Any],
    split: str,
) -> FeatureDataset:
    part = frame[frame["split"] == split].copy()
    feature_indices = torch.tensor(part["feature_index"].to_numpy(), dtype=torch.long)
    features = feature_cache["features"].float().index_select(0, feature_indices)
    labels = torch.tensor(part["train_label_id"].to_numpy(), dtype=torch.long)
    crop_ids = torch.tensor(part["crop_id"].to_numpy(), dtype=torch.long)
    return FeatureDataset(features, labels, crop_ids)


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str,
) -> dict[str, Any]:
    model.eval()
    losses: list[torch.Tensor] = []
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    crop_ids: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    with torch.inference_mode():
        for x_cpu, y_cpu, crop_ids_cpu in tqdm(loader, desc=desc, leave=False):
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            logits = model(x)
            sample_loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
            prob = torch.softmax(logits, dim=1)
            confidence, pred = prob.max(dim=1)
            losses.append(sample_loss.detach().cpu())
            preds.append(pred.detach().cpu())
            labels.append(y_cpu.detach().cpu())
            crop_ids.append(crop_ids_cpu.detach().cpu())
            confidences.append(confidence.detach().cpu())

    sample_loss_np = torch.cat(losses).numpy()
    preds_np = torch.cat(preds).numpy()
    labels_np = torch.cat(labels).numpy()
    return {
        "loss": float(sample_loss_np.mean()),
        "accuracy": float(accuracy_score(labels_np, preds_np)),
        "sample_loss": sample_loss_np,
        "preds": preds_np,
        "labels": labels_np,
        "crop_ids": torch.cat(crop_ids).numpy(),
        "confidence": torch.cat(confidences).numpy(),
    }


def build_prediction_frame(
    metrics: dict[str, Any],
    split_frame_data: pd.DataFrame,
    id_to_label: dict[int, str],
    split: str,
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "crop_id": metrics["crop_ids"].astype(int),
            "label_id": metrics["labels"].astype(int),
            "pred_id": metrics["preds"].astype(int),
            "confidence": metrics["confidence"].astype(float),
            "loss": metrics["sample_loss"].astype(float),
        }
    )
    predictions["label"] = predictions["label_id"].map(id_to_label)
    predictions["pred_label"] = predictions["pred_id"].map(id_to_label)
    predictions["correct"] = predictions["label_id"] == predictions["pred_id"]
    predictions["split"] = split
    meta_columns = [
        "crop_id",
        "image_path",
        "manual_reviewed",
        "series",
        "fine_grained_series",
        "submodel",
        "operator_en",
        "operator_jp",
        "power_type",
    ]
    meta_columns = [column for column in meta_columns if column in split_frame_data.columns]
    return predictions.merge(split_frame_data[meta_columns], on="crop_id", how="left")


def save_classification_report(
    metrics: dict[str, Any],
    id_to_label: dict[int, str],
    output_path: Path,
) -> None:
    labels = list(range(len(id_to_label)))
    target_names = [id_to_label[idx] for idx in labels]
    report = classification_report(
        metrics["labels"],
        metrics["preds"],
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report).T.to_csv(output_path, encoding="utf-8")


def train(config: dict[str, Any], train_config: dict[str, Any], run_name: str | None = None) -> Path:
    set_seed(int(train_config["seed"]))
    metadata = load_dataset_metadata(config, train_config)
    feature_cache_path = resolve_feature_cache_path(config, train_config)
    feature_cache = load_feature_cache(feature_cache_path)
    frame, label_to_id, id_to_label, data_report = build_training_frame(
        metadata=metadata,
        feature_cache=feature_cache,
        train_config=train_config,
    )
    frame = split_frame(frame, train_config)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = run_name or f"linear_probe_{timestamp}"
    run_root = utils.join_data_root(train_config["run_dir"], config=config)
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    split_columns = [
        "crop_id",
        "split",
        "image_path",
        train_config["label_column"],
        "train_label_id",
        "feature_index",
        "manual_reviewed",
    ]
    split_columns = [column for column in split_columns if column in frame.columns]
    frame[split_columns].to_csv(run_dir / "splits.csv", index=False, encoding="utf-8")
    write_json(run_dir / "label_to_id.json", label_to_id)
    write_json(run_dir / "id_to_label.json", {str(key): value for key, value in id_to_label.items()})
    write_json(
        run_dir / "config_snapshot.json",
        {
            "training_config": train_config,
            "feature_cache_path": str(feature_cache_path),
            "data_report": data_report,
            "split_counts": frame["split"].value_counts().to_dict(),
        },
    )

    train_ds = make_split_dataset(frame, feature_cache, "train")
    val_ds = make_split_dataset(frame, feature_cache, "val")
    test_ds = make_split_dataset(frame, feature_cache, "test")
    batch_size = int(train_config["batch_size"])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = LinearProbe(
        input_dim=int(feature_cache["features"].shape[1]),
        num_classes=len(label_to_id),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    max_epochs = int(train_config["max_epochs"])
    patience = int(train_config["patience"])
    min_delta = float(train_config["min_delta"])

    print(
        f"run_dir={run_dir}\n"
        f"feature_cache={feature_cache_path}\n"
        f"samples={data_report['trainable_rows']} classes={data_report['num_classes']} device={device}"
    )
    for epoch in tqdm(range(max_epochs), desc="linear probe epochs"):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_n = 0
        for x_cpu, y_cpu, _ in train_loader:
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * int(y.numel())
            running_correct += int(logits.argmax(dim=1).eq(y).sum().detach().cpu())
            running_n += int(y.numel())

        val_metrics = evaluate(model, val_loader, criterion, device, desc=f"epoch {epoch} val")
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, running_n),
            "train_accuracy": running_correct / max(1, running_n),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(row)

        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping at epoch={epoch}; best_epoch={best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint")
    model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False, encoding="utf-8")

    val_metrics = evaluate(model, val_loader, criterion, device, desc="final val")
    test_metrics = evaluate(model, test_loader, criterion, device, desc="final test")
    metrics_summary = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "val_loss": float(val_metrics["loss"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "num_classes": int(len(label_to_id)),
        "train_samples": int(len(train_ds)),
        "val_samples": int(len(val_ds)),
        "test_samples": int(len(test_ds)),
    }
    write_json(run_dir / "metrics.json", metrics_summary)
    torch.save(
        {
            "model_state_dict": best_state,
            "input_dim": int(feature_cache["features"].shape[1]),
            "num_classes": len(label_to_id),
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "metrics": metrics_summary,
            "training_config": copy.deepcopy(train_config),
            "feature_cache_path": str(feature_cache_path),
        },
        run_dir / "best_linear_probe.pt",
    )

    build_prediction_frame(val_metrics, frame[frame["split"] == "val"], id_to_label, "val").to_csv(
        run_dir / "val_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    build_prediction_frame(test_metrics, frame[frame["split"] == "test"], id_to_label, "test").to_csv(
        run_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8",
    )
    save_classification_report(val_metrics, id_to_label, run_dir / "val_classification_report.csv")
    save_classification_report(test_metrics, id_to_label, run_dir / "test_classification_report.csv")
    write_json(run_root / "latest_linear_probe.json", {"run_name": run_name, "run_dir": str(run_dir)})
    print(json.dumps(metrics_summary, ensure_ascii=False, indent=2))
    return run_dir


def main() -> None:
    args = parse_args()
    config = utils.load_pipeline_config(args.config)
    train_config = copy.deepcopy(config["training"]["linear_probe"])
    if args.max_epochs is not None:
        train_config["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        train_config["batch_size"] = args.batch_size
    if args.feature_cache_file is not None:
        train_config["feature_cache_file"] = args.feature_cache_file
    train(config, train_config, run_name=args.run_name)


if __name__ == "__main__":
    main()
