import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoImageProcessor

from pipeline import utils
from trainer.checkpoint import save_checkpoint, write_json
from trainer.dataset import CropCollator, CropDataset
from trainer.eval import build_eval_report, evaluate
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
    num_workers = int(trainer_cfg["num_workers"])
    dataloader_kwargs = {
        "batch_size": int(trainer_cfg["batch_size"]),
        "shuffle": train,
        "num_workers": num_workers,
        "collate_fn": CropCollator(processor),
        "pin_memory": bool(trainer_cfg["pin_memory"]),
        "drop_last": train and bool(trainer_cfg["drop_last"]),
    }
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = bool(trainer_cfg["persistent_workers"])
        dataloader_kwargs["prefetch_factor"] = int(trainer_cfg["prefetch_factor"])

    dataset = CropDataset(
        metadata=metadata,
        dataset_root=dataset_root,
        image_path_column=trainer_cfg["image_path_column"],
        label_id_column=trainer_cfg["label_id_column"],
    )
    return DataLoader(dataset, **dataloader_kwargs)


def make_feature_cache_path(
    *,
    config: dict[str, Any],
    trainer_cfg: dict[str, Any],
    feature_cache_file_name: str,
) -> Path:
    feature_cache_dir = utils.join_data_root(trainer_cfg["feature_cache_dir"], config=config)
    return feature_cache_dir / feature_cache_file_name


@torch.inference_mode()
def extract_feature_table(
    *,
    model: BackboneLinearClassifier,
    dataloader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    features = []
    labels = []
    sample_indices = []
    image_paths = []
    for batch in tqdm(dataloader, desc="extract features", unit="batch"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            batch_features = model.extract_features(pixel_values)
        features.append(batch_features.float().cpu())
        labels.append(batch["labels"].cpu())
        sample_indices.append(batch["sample_index"].cpu())
        image_paths.extend(batch["image_path"])
    return {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "sample_indices": torch.cat(sample_indices, dim=0),
        "image_paths": image_paths,
    }


def load_or_create_feature_cache(
    *,
    config: dict[str, Any],
    trainer_cfg: dict[str, Any],
    phase_cfg: dict[str, Any],
    model: BackboneLinearClassifier,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset_root: Path,
    processor: Any,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    feature_cache_path = make_feature_cache_path(
        config=config,
        trainer_cfg=trainer_cfg,
        feature_cache_file_name=phase_cfg["feature_cache_file_name"],
    )
    if feature_cache_path.exists() and not bool(phase_cfg["feature_cache_rebuild"]):
        logger.info("加载linear head特征缓存: %s", feature_cache_path)
        cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
        if cache["backbone_model_name"] != trainer_cfg["backbone_model_name"]:
            raise ValueError(
                "linear head特征缓存的backbone_model_name与当前配置不一致，"
                "请设置feature_cache_rebuild=true后重建。"
            )
        if len(cache["train"]["labels"]) != len(train_df) or len(cache["val"]["labels"]) != len(val_df):
            raise ValueError(
                "linear head特征缓存样本数与当前train/val切分不一致，"
                "请设置feature_cache_rebuild=true后重建。"
            )
        image_path_column = trainer_cfg["image_path_column"]
        train_paths = [str(path).replace("\\", "/") for path in train_df[image_path_column].tolist()]
        val_paths = [str(path).replace("\\", "/") for path in val_df[image_path_column].tolist()]
        if cache["train"]["image_paths"] != train_paths or cache["val"]["image_paths"] != val_paths:
            raise ValueError(
                "linear head特征缓存的image_path顺序与当前train/val切分不一致，"
                "请设置feature_cache_rebuild=true后重建。"
            )
        validate_feature_cache(cache)
        return cache

    logger.info("开始生成linear head特征缓存: %s", feature_cache_path)
    feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
    train_feature_loader = make_dataloader(
        metadata=train_df,
        dataset_root=dataset_root,
        processor=processor,
        trainer_cfg=trainer_cfg,
        train=False,
    )
    val_feature_loader = make_dataloader(
        metadata=val_df,
        dataset_root=dataset_root,
        processor=processor,
        trainer_cfg=trainer_cfg,
        train=False,
    )
    model.train_linear_head_only()
    model.to(device)
    cache = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "backbone_model_name": trainer_cfg["backbone_model_name"],
        "train_sample_count": int(len(train_df)),
        "val_sample_count": int(len(val_df)),
        "train": extract_feature_table(
            model=model,
            dataloader=train_feature_loader,
            device=device,
            amp_enabled=bool(phase_cfg["feature_cache_amp_enabled"]) and amp_enabled,
            amp_dtype=amp_dtype,
        ),
        "val": extract_feature_table(
            model=model,
            dataloader=val_feature_loader,
            device=device,
            amp_enabled=bool(phase_cfg["feature_cache_amp_enabled"]) and amp_enabled,
            amp_dtype=amp_dtype,
        ),
    }
    validate_feature_cache(cache)
    torch.save(cache, feature_cache_path)
    logger.info("linear head特征缓存已保存: %s", feature_cache_path)
    return cache


def validate_feature_cache(cache: dict[str, Any]) -> None:
    for split in ("train", "val"):
        features = cache[split]["features"]
        labels = cache[split]["labels"]
        if not torch.isfinite(features).all():
            nan_count = int(torch.isnan(features).sum().item())
            inf_count = int(torch.isinf(features).sum().item())
            raise ValueError(
                f"linear head特征缓存包含非有限feature: split={split}, "
                f"nan={nan_count}, inf={inf_count}。请删除缓存或设置feature_cache_rebuild=true后重建。"
            )
        if labels.numel() == 0:
            raise ValueError(f"linear head特征缓存为空: split={split}")


def make_feature_dataloader(
    *,
    feature_table: dict[str, Any],
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        feature_table["features"],
        feature_table["labels"].long(),
        feature_table["sample_indices"].long(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float | int]:
    model.train()
    loss_chunks = []
    correct_chunks = []
    sample_count = 0
    for batch in tqdm(dataloader, desc="train", unit="batch"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(pixel_values)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("训练中出现非有限loss，请检查输入、AMP和学习率。")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_size = int(labels.numel())
            loss_chunks.append(loss.detach() * batch_size)
            correct_chunks.append(preds.eq(labels).sum().detach())
            sample_count += batch_size

    loss_sum = torch.stack(loss_chunks).sum().item() if loss_chunks else 0.0
    correct_count = torch.stack(correct_chunks).sum().item() if correct_chunks else 0
    return {
        "loss": loss_sum / max(1, sample_count),
        "accuracy": correct_count / max(1, sample_count),
        "n": sample_count,
    }


def train_feature_head_one_epoch(
    *,
    model: BackboneLinearClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float | int]:
    model.classifier.train()
    loss_chunks = []
    correct_chunks = []
    sample_count = 0
    for features_cpu, labels_cpu, _sample_indices_cpu in tqdm(dataloader, desc="train features", unit="batch"):
        features = features_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model.classifier(features)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("feature linear head训练中出现非有限loss，请重建特征缓存或关闭AMP。")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_size = int(labels.numel())
            loss_chunks.append(loss.detach() * batch_size)
            correct_chunks.append(preds.eq(labels).sum().detach())
            sample_count += batch_size

    loss_sum = torch.stack(loss_chunks).sum().item() if loss_chunks else 0.0
    correct_count = torch.stack(correct_chunks).sum().item() if correct_chunks else 0
    return {
        "loss": loss_sum / max(1, sample_count),
        "accuracy": correct_count / max(1, sample_count),
        "n": sample_count,
    }


@torch.inference_mode()
def evaluate_feature_head(
    *,
    model: BackboneLinearClassifier,
    dataloader: DataLoader,
    feature_table: dict[str, Any],
    labels: pd.DataFrame,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.classifier.eval()
    records = []
    image_paths = feature_table["image_paths"]
    for features_cpu, labels_cpu, sample_indices_cpu in tqdm(dataloader, desc="eval features", unit="batch"):
        features = features_cpu.to(device, non_blocking=True)
        y_true = labels_cpu.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model.classifier(features)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("feature linear head验证中出现非有限logits，请重建特征缓存或降低学习率。")
        probs = torch.softmax(logits, dim=1)
        confidence, y_pred = probs.max(dim=1)
        for i, sample_index in enumerate(sample_indices_cpu.tolist()):
            records.append(
                {
                    "sample_index": int(sample_index),
                    "image_path": image_paths[int(sample_index)],
                    "label_id": int(y_true[i].item()),
                    "pred_id": int(y_pred[i].item()),
                    "pred_confidence": float(confidence[i].item()),
                    "correct": bool(y_pred[i].eq(y_true[i]).item()),
                }
            )
    predictions = pd.DataFrame(records)
    return build_eval_report(predictions=predictions, labels=labels), predictions


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


def get_amp_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError("trainer.amp_dtype必须是float16或bfloat16")


def prepare_phase(model: BackboneLinearClassifier, phase_cfg: dict[str, Any]) -> None:
    train_mode = phase_cfg["train_mode"]
    if train_mode == "linear_head":
        model.train_linear_head_only()
        return
    if train_mode == "lora":
        model.enable_lora(
            r=int(phase_cfg["lora_r"]),
            alpha=int(phase_cfg["lora_alpha"]),
            dropout=float(phase_cfg["lora_dropout"]),
            bias=str(phase_cfg["lora_bias"]),
        )
        model.train_lora_and_head()
        return
    raise ValueError("trainer.phases[].train_mode必须是linear_head或lora")


def make_phase_optimizer(
    *,
    model: BackboneLinearClassifier,
    phase_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    train_mode = phase_cfg["train_mode"]
    weight_decay = float(phase_cfg["weight_decay"])
    if train_mode == "linear_head":
        return torch.optim.AdamW(
            model.classifier.parameters(),
            lr=float(phase_cfg["learning_rate"]),
            weight_decay=weight_decay,
        )
    if train_mode == "lora":
        lora_parameters = [
            parameter
            for name, parameter in model.backbone.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        if not lora_parameters:
            raise ValueError("LoRA phase没有可训练的LoRA参数")
        return torch.optim.AdamW(
            [
                {
                    "params": model.classifier.parameters(),
                    "lr": float(phase_cfg["head_learning_rate"]),
                },
                {
                    "params": lora_parameters,
                    "lr": float(phase_cfg["lora_learning_rate"]),
                },
            ],
            weight_decay=weight_decay,
        )
    raise ValueError(f"未知训练模式: {train_mode!r}")


def run_phase(
    *,
    phase_cfg: dict[str, Any],
    model: BackboneLinearClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    dataset_root: Path,
    processor: Any,
    labels: pd.DataFrame,
    criterion: nn.Module,
    device: torch.device,
    trainer_cfg: dict[str, Any],
    config: dict[str, Any],
    run_dir: Path,
    labels_payload: list[dict[str, Any]],
    epoch_rows: list[dict[str, Any]],
    global_epoch_start: int,
) -> tuple[int, dict[str, Any]]:
    prepare_phase(model, phase_cfg)
    model.to(device)
    optimizer = make_phase_optimizer(
        model=model,
        phase_cfg=phase_cfg,
    )

    phase_name = str(phase_cfg["name"])
    amp_enabled = bool(trainer_cfg["amp_enabled"]) and device.type == "cuda"
    amp_dtype = get_amp_dtype(str(trainer_cfg["amp_dtype"]))
    use_feature_cache = bool(phase_cfg["use_feature_cache"])
    early_stopping_enabled = bool(phase_cfg["early_stopping_enabled"])
    early_stopping_monitor = str(phase_cfg["early_stopping_monitor"])
    early_stopping_mode = str(phase_cfg["early_stopping_mode"])
    early_stopping_patience = int(phase_cfg["early_stopping_patience"])
    early_stopping_min_delta = float(phase_cfg["early_stopping_min_delta"])
    if early_stopping_patience < 1:
        raise ValueError("trainer.phases[].early_stopping_patience必须大于等于1")

    best_score = None
    best_checkpoint_path = None
    best_epoch = None
    epochs_without_improvement = 0
    stopped_early = False
    completed_epochs = 0
    feature_cache = None
    feature_train_loader = None
    feature_val_loader = None
    if use_feature_cache:
        if phase_cfg["train_mode"] != "linear_head":
            raise ValueError("feature cache目前只支持linear_head phase")
        feature_cache = load_or_create_feature_cache(
            config=config,
            trainer_cfg=trainer_cfg,
            phase_cfg=phase_cfg,
            model=model,
            train_df=train_df,
            val_df=val_df,
            dataset_root=dataset_root,
            processor=processor,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        feature_train_loader = make_feature_dataloader(
            feature_table=feature_cache["train"],
            batch_size=int(trainer_cfg["batch_size"]),
            shuffle=True,
            drop_last=bool(trainer_cfg["drop_last"]),
            pin_memory=bool(trainer_cfg["pin_memory"]),
        )
        feature_val_loader = make_feature_dataloader(
            feature_table=feature_cache["val"],
            batch_size=int(trainer_cfg["batch_size"]),
            shuffle=False,
            drop_last=False,
            pin_memory=bool(trainer_cfg["pin_memory"]),
        )

    logger.info("开始phase=%s, mode=%s, epochs=%d", phase_name, phase_cfg["train_mode"], int(phase_cfg["epochs"]))
    for phase_epoch in range(int(phase_cfg["epochs"])):
        global_epoch = global_epoch_start + phase_epoch
        if use_feature_cache:
            train_metrics = train_feature_head_one_epoch(
                model=model,
                dataloader=feature_train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                amp_enabled=False,
                amp_dtype=amp_dtype,
            )
            eval_report, predictions = evaluate_feature_head(
                model=model,
                dataloader=feature_val_loader,
                feature_table=feature_cache["val"],
                labels=labels,
                device=device,
                amp_enabled=False,
                amp_dtype=amp_dtype,
            )
        else:
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            eval_report, predictions = evaluate(
                model=model,
                dataloader=val_loader,
                labels=labels,
                device=device,
            )
        epoch_row = {
            "phase": phase_name,
            "phase_epoch": phase_epoch,
            "epoch": global_epoch,
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

        checkpoint_path = run_dir / (
            f"{trainer_cfg['checkpoint_prefix']}_{phase_name}_epoch{phase_epoch:03d}.pt"
        )
        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=global_epoch,
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
            best_epoch = global_epoch
            epochs_without_improvement = 0
            best_checkpoint_path = run_dir / f"{trainer_cfg['checkpoint_prefix']}_{phase_name}_best.pt"
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=global_epoch,
                config=trainer_cfg,
                metrics=epoch_row,
                labels=labels_payload,
            )
        else:
            epochs_without_improvement += 1
        completed_epochs += 1
        logger.info(
            "phase=%s epoch=%d train_loss=%.4f train_acc=%.4f val_acc=%.4f val_macro_f1=%.4f best_%s=%.4f stale_epochs=%d",
            phase_name,
            phase_epoch,
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
                "phase=%s early stopping触发: monitor=%s, mode=%s, patience=%d, best_epoch=%s, best_score=%.4f",
                phase_name,
                early_stopping_monitor,
                early_stopping_mode,
                early_stopping_patience,
                best_epoch,
                float(best_score) if best_score is not None else float("nan"),
            )
            break

    return completed_epochs, {
        "phase": phase_name,
        "train_mode": phase_cfg["train_mode"],
        "best_score": best_score,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else None,
        "early_stopping_monitor": early_stopping_monitor,
        "early_stopping_mode": early_stopping_mode,
        "stopped_early": stopped_early,
        "completed_epochs": completed_epochs,
        "use_feature_cache": use_feature_cache,
    }


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
    phase_summaries = []
    global_epoch = 0
    for phase_cfg in trainer_cfg["phases"]:
        completed_epochs, phase_summary = run_phase(
            phase_cfg=phase_cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_df=train_df,
            val_df=val_df,
            dataset_root=dataset_root,
            processor=processor,
            labels=labels,
            criterion=criterion,
            device=device,
            trainer_cfg=trainer_cfg,
            config=config,
            run_dir=run_dir,
            labels_payload=labels_payload,
            epoch_rows=epoch_rows,
            global_epoch_start=global_epoch,
        )
        global_epoch += completed_epochs
        phase_summaries.append(phase_summary)

    write_json(
        run_dir / "run_summary.json",
        {
            "run_dir": str(run_dir),
            "phase_summaries": phase_summaries,
            "total_completed_epochs": global_epoch,
            "num_train_samples": int(len(train_df)),
            "num_val_samples": int(len(val_df)),
            "num_classes": int(len(labels)),
        },
    )


if __name__ == "__main__":
    main()
