from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@torch.inference_mode()
def evaluate(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    labels: pd.DataFrame,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    records = []
    for batch in tqdm(dataloader, desc="eval", unit="batch"):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        y_true = batch["labels"].to(device, non_blocking=True)
        logits = model(pixel_values)
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
    return build_eval_report(predictions=predictions, labels=labels), predictions


def build_eval_report(
    *,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
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
    return report
