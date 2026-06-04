from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def validate_top_k_values(top_k_values: list[int], num_classes: int) -> list[int]:
    normalized_values = sorted({int(top_k) for top_k in top_k_values})
    if not normalized_values:
        raise ValueError("trainer.top_k至少需要包含一个正整数")
    invalid_values = [top_k for top_k in normalized_values if top_k < 1]
    if invalid_values:
        raise ValueError(f"trainer.top_k只能包含正整数: {invalid_values}")
    too_large_values = [top_k for top_k in normalized_values if top_k > num_classes]
    if too_large_values:
        raise ValueError(
            f"trainer.top_k不能超过类别数num_classes={num_classes}: {too_large_values}"
        )
    return normalized_values


def make_top_k_correct_counts(top_k_values: list[int]) -> dict[int, int]:
    return {top_k: 0 for top_k in top_k_values}


def update_top_k_correct_counts(
    *,
    counts: dict[int, int],
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if not counts or labels.numel() == 0:
        return
    max_top_k = max(counts)
    top_k_pred_ids = logits.topk(k=max_top_k, dim=1).indices
    top_k_matches = top_k_pred_ids.eq(labels.unsqueeze(1))
    for top_k in counts:
        counts[top_k] += int(top_k_matches[:, :top_k].any(dim=1).sum().item())


def build_top_k_accuracy(
    *,
    counts: dict[int, int],
    sample_count: int,
) -> dict[str, float]:
    return {
        f"top_{top_k}_accuracy": correct_count / max(1, sample_count)
        for top_k, correct_count in counts.items()
    }


@torch.inference_mode()
def evaluate(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    labels: pd.DataFrame,
    device: torch.device,
    top_k_values: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    records = []
    top_k_correct_counts = make_top_k_correct_counts(top_k_values)
    for batch in tqdm(dataloader, desc="eval", unit="batch"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        y_true = batch["labels"].to(device, non_blocking=True)
        logits = model(pixel_values)
        update_top_k_correct_counts(
            counts=top_k_correct_counts,
            logits=logits,
            labels=y_true,
        )
        probs = torch.softmax(logits, dim=1)
        confidence, y_pred = probs.max(dim=1)
        for i, image_path in enumerate(batch["image_path"]):
            records.append(
                {
                    "sample_index": int(batch["sample_index"][i].item()),
                    "image_path": image_path,
                    "label_id": int(y_true[i].item()),
                    "pred_id": int(y_pred[i].item()),
                    "pred_confidence": float(confidence[i].item()),
                    "correct": bool(y_pred[i].eq(y_true[i]).item()),
                }
            )

    predictions = pd.DataFrame(records)
    return (
        build_eval_report(
            predictions=predictions,
            labels=labels,
            top_k_accuracy=build_top_k_accuracy(
                counts=top_k_correct_counts,
                sample_count=len(predictions),
            ),
        ),
        predictions,
    )


def build_eval_report(
    *,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    top_k_accuracy: dict[str, float] | None = None,
) -> dict[str, Any]:
    label_ids = labels["label_id"].astype(int).tolist()
    id_to_label = {
        int(row["label_id"]): str(row["label"])
        for _, row in labels.iterrows()
    }
    y_true_np = predictions["label_id"].astype(int).to_numpy()
    y_pred_np = predictions["pred_id"].astype(int).to_numpy()
    report = {
        "num_samples": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(
            f1_score(
                y_true_np,
                y_pred_np,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true_np,
                y_pred_np,
                labels=label_ids,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_true_np,
            y_pred_np,
            labels=label_ids,
            target_names=[id_to_label[label_id] for label_id in label_ids],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": label_ids,
            "matrix": confusion_matrix(y_true_np, y_pred_np, labels=label_ids).astype(int).tolist(),
        },
    }
    if top_k_accuracy is not None:
        report["top_k_accuracy"] = {
            key: float(value)
            for key, value in top_k_accuracy.items()
        }
    return report
