import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

from pipeline import utils
from trainer.checkpoint import save_checkpoint, write_json
from trainer.dataset import CropCollator, CropDataset
from trainer.eval import evaluate
from trainer.model import BackboneLinearClassifier

logger = utils.get_logger("trainer")


def get_torch_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tables(config: dict[str, Any]) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    trainer_cfg = config["trainer"]
    dataset_root = utils.join_data_root(config["path"]["dataset_dir"], config=config)
    metadata_path = dataset_root / trainer_cfg["metadata_file_name"]
    labels_path = dataset_root / trainer_cfg["labels_file_name"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata文件不存在: {metadata_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels文件不存在: {labels_path}")

    metadata = pd.read_csv(metadata_path)
    labels = pd.read_csv(labels_path)
    validate_tables(metadata=metadata, labels=labels, trainer_cfg=trainer_cfg)
    return dataset_root, metadata, labels


def validate_tables(
    *,
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    trainer_cfg: dict[str, Any],
) -> None:
    image_path_column = trainer_cfg["image_path_column"]
    label_id_column = trainer_cfg["label_id_column"]
    missing_metadata_columns = {image_path_column, label_id_column} - set(metadata.columns)
    if missing_metadata_columns:
        raise ValueError(f"metadata缺少必要列: {sorted(missing_metadata_columns)}")
    if {"label_id", "label"} - set(labels.columns):
        raise ValueError("labels.csv必须包含label_id和label列")
    label_ids = sorted(labels["label_id"].astype(int).tolist())
    if label_ids != list(range(len(label_ids))):
        raise ValueError("labels.csv中的label_id必须从0开始连续编号")
    metadata_label_ids = set(metadata[label_id_column].dropna().astype(int).tolist())
    missing_label_ids = metadata_label_ids - set(label_ids)
    if missing_label_ids:
        raise ValueError(f"metadata中存在labels.csv没有定义的label_id: {sorted(missing_label_ids)}")


def split_metadata(
    *,
    metadata: pd.DataFrame,
    trainer_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_id_column = trainer_cfg["label_id_column"]
    val_ratio = float(trainer_cfg["val_ratio"])
    if not 0 < val_ratio < 1:
        raise ValueError("trainer.val_ratio必须在0和1之间")

    stratify = None
    if bool(trainer_cfg["stratify_split"]):
        counts = metadata[label_id_column].value_counts()
        if int(counts.min()) >= 2:
            stratify = metadata[label_id_column]
        else:
            logger.warning("部分标签样本数小于2，改用随机切分。")

    train_df, val_df = train_test_split(
        metadata,
        test_size=val_ratio,
        random_state=int(trainer_cfg["seed"]),
        shuffle=True,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def make_dataloader(
    *,
    metadata: pd.DataFrame,
    dataset_root: Path,
    processor: Any,
    trainer_cfg: dict[str, Any],
    train: bool,
) -> DataLoader:
    dataset = CropDataset(
        metadata=metadata,
        dataset_root=dataset_root,
        image_path_column=trainer_cfg["image_path_column"],
        label_id_column=trainer_cfg["label_id_column"],
    )
    return DataLoader(
        dataset,
        batch_size=int(trainer_cfg["batch_size"]),
        shuffle=train,
        num_workers=int(trainer_cfg["num_workers"]),
        collate_fn=CropCollator(processor),
        pin_memory=bool(trainer_cfg["pin_memory"]),
    )


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float | int]:
    model.train()
    loss_sum = 0.0
    correct_count = 0
    sample_count = 0
    for batch in tqdm(dataloader, desc="train", unit="batch"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_size = int(labels.numel())
            loss_sum += float(loss.item()) * batch_size
            correct_count += int(preds.eq(labels).sum().item())
            sample_count += batch_size

    return {
        "loss": loss_sum / max(1, sample_count),
        "accuracy": correct_count / max(1, sample_count),
        "n": sample_count,
    }


def is_metric_improved(
    *,
    current_value: float,
    best_value: float | None,
    mode: str,
    min_delta: float,
) -> bool:
    if best_value is None:
        return True
    if mode == "max":
        return current_value > best_value + min_delta
    if mode == "min":
        return current_value < best_value - min_delta
    raise ValueError("trainer.early_stopping_mode必须是'max'或'min'")


def main(config: dict[str, Any] | None = None) -> None:
    if config is None:
        config = utils.load_pipeline_config()

    trainer_cfg = config["trainer"]
    set_seed(int(trainer_cfg["seed"]))
    dataset_root, metadata, labels = load_tables(config)
    train_df, val_df = split_metadata(metadata=metadata, trainer_cfg=trainer_cfg)
    device = get_torch_device(trainer_cfg["device"])

    processor = AutoImageProcessor.from_pretrained(trainer_cfg["backbone_model_name"])
    train_loader = make_dataloader(
        metadata=train_df,
        dataset_root=dataset_root,
        processor=processor,
        trainer_cfg=trainer_cfg,
        train=True,
    )
    val_loader = make_dataloader(
        metadata=val_df,
        dataset_root=dataset_root,
        processor=processor,
        trainer_cfg=trainer_cfg,
        train=False,
    )

    model = BackboneLinearClassifier(
        backbone_model_name=trainer_cfg["backbone_model_name"],
        num_classes=int(labels["label_id"].astype(int).max()) + 1,
        freeze_backbone=bool(trainer_cfg["freeze_backbone"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(trainer_cfg["learning_rate"]),
        weight_decay=float(trainer_cfg["weight_decay"]),
    )

    run_dir = utils.join_data_root(trainer_cfg["output_dir"], config=config) / time.strftime(
        "%Y%m%d_%H%M%S",
        time.localtime(),
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    logger.info(
        "开始训练: train=%d, val=%d, labels=%d, device=%s, run_dir=%s",
        len(train_df),
        len(val_df),
        len(labels),
        device,
        run_dir,
    )

    labels_payload = labels.to_dict(orient="records")
    epoch_rows = []
    early_stopping_enabled = bool(trainer_cfg["early_stopping_enabled"])
    early_stopping_monitor = str(trainer_cfg["early_stopping_monitor"])
    early_stopping_mode = str(trainer_cfg["early_stopping_mode"])
    early_stopping_patience = int(trainer_cfg["early_stopping_patience"])
    early_stopping_min_delta = float(trainer_cfg["early_stopping_min_delta"])
    if early_stopping_patience < 1:
        raise ValueError("trainer.early_stopping_patience必须大于等于1")

    best_score = None
    best_checkpoint_path = None
    best_epoch = None
    epochs_without_improvement = 0
    stopped_early = False
    for epoch in range(int(trainer_cfg["epochs"])):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        eval_report, predictions = evaluate(
            model=model,
            dataloader=val_loader,
            labels=labels,
            device=device,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": eval_report["accuracy"],
            "val_macro_f1": eval_report["macro_f1"],
            "val_weighted_f1": eval_report["weighted_f1"],
            "train_n": train_metrics["n"],
            "val_n": eval_report["num_samples"],
        }
        if early_stopping_monitor not in epoch_row:
            raise ValueError(f"early stopping监控指标不存在: {early_stopping_monitor!r}")
        epoch_rows.append(epoch_row)
        pd.DataFrame(epoch_rows).to_csv(run_dir / trainer_cfg["epoch_report_file_name"], index=False)
        predictions.to_csv(run_dir / trainer_cfg["prediction_file_name"], index=False)
        write_json(run_dir / trainer_cfg["eval_report_file_name"], eval_report)

        checkpoint_path = run_dir / f"{trainer_cfg['checkpoint_prefix']}_epoch{epoch:03d}.pt"
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            config=trainer_cfg,
            metrics=epoch_row,
            labels=labels_payload,
        )
        monitor_value = float(epoch_row[early_stopping_monitor])
        if is_metric_improved(
            current_value=monitor_value,
            best_value=best_score,
            mode=early_stopping_mode,
            min_delta=early_stopping_min_delta,
        ):
            best_score = monitor_value
            best_epoch = epoch
            epochs_without_improvement = 0
            best_checkpoint_path = run_dir / f"{trainer_cfg['checkpoint_prefix']}_best.pt"
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=trainer_cfg,
                metrics=epoch_row,
                labels=labels_payload,
            )
        else:
            epochs_without_improvement += 1
        logger.info(
            "epoch=%d train_loss=%.4f train_acc=%.4f val_acc=%.4f val_macro_f1=%.4f best_%s=%.4f stale_epochs=%d",
            epoch,
            float(train_metrics["loss"]),
            float(train_metrics["accuracy"]),
            float(eval_report["accuracy"]),
            float(eval_report["macro_f1"]),
            early_stopping_monitor,
            float(best_score) if best_score is not None else float("nan"),
            epochs_without_improvement,
        )
        if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            logger.info(
                "early stopping触发: monitor=%s, mode=%s, patience=%d, best_epoch=%s, best_score=%.4f",
                early_stopping_monitor,
                early_stopping_mode,
                early_stopping_patience,
                best_epoch,
                float(best_score) if best_score is not None else float("nan"),
            )
            break

    write_json(
        run_dir / "run_summary.json",
        {
            "run_dir": str(run_dir),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "early_stopping_monitor": early_stopping_monitor,
            "early_stopping_mode": early_stopping_mode,
            "stopped_early": stopped_early,
            "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
            "num_train_samples": int(len(train_df)),
            "num_val_samples": int(len(val_df)),
            "num_classes": int(len(labels)),
        },
    )


if __name__ == "__main__":
    main()
