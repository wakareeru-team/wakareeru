import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoImageProcessor

from model_core.model import BackboneLinearClassifier


@dataclass(frozen=True)
class LoadedClassifier:
    model: BackboneLinearClassifier
    processor: Any
    model_config: dict[str, Any]
    labels: list[dict[str, Any]]
    id_to_label: dict[int, str]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_id_to_label(labels: list[dict[str, Any]]) -> dict[int, str]:
    return {
        int(row["label_id"]): str(row["label"])
        for row in labels
    }


def load_classifier(
    model_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    local_files_only: bool = True,
) -> LoadedClassifier:
    model_dir = Path(model_dir)
    model_config = read_json(model_dir / "model_config.json")
    labels = read_json(model_dir / "labels.json")

    model = BackboneLinearClassifier(
        backbone_model_name=str(model_config["backbone_model_name"]),
        num_classes=int(model_config["num_classes"]),
        freeze_backbone=True,
    )
    classifier_state = load_file(model_dir / "classifier.safetensors")
    model.classifier.load_state_dict(classifier_state)
    model.to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(
        model_dir / "processor",
        local_files_only=local_files_only,
    )

    return LoadedClassifier(
        model=model,
        processor=processor,
        model_config=model_config,
        labels=labels,
        id_to_label=build_id_to_label(labels),
    )
