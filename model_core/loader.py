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
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_id_to_label(labels: list[dict[str, Any]]) -> dict[int, str]:
    return {
        int(row["label_id"]): str(row["label"])
        for row in labels
    }


def resolve_backbone_path(model_dir: Path, model_config: dict[str, Any]) -> Path:
    backbone_config = model_config.get("backbone")
    if not isinstance(backbone_config, dict) or not backbone_config.get("path"):
        raise ValueError("model_config.json is missing backbone.path")
    backbone_path = Path(str(backbone_config["path"]))
    if not backbone_path.is_absolute():
        backbone_path = model_dir / backbone_path
    if not backbone_path.is_dir():
        raise FileNotFoundError(f"Backbone directory not found: {backbone_path}")
    return backbone_path


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def require_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_classifier(
    model_dir: str | Path,
    *,
    device: torch.device | str = "cpu",
    local_files_only: bool = True,
) -> LoadedClassifier:
    model_dir = Path(model_dir)
    require_dir(model_dir, "Model directory")
    model_config = read_json(model_dir / "model_config.json")
    labels = read_json(model_dir / "labels.json")
    backbone_path = resolve_backbone_path(model_dir, model_config)
    processor_path = require_dir(model_dir / "processor", "Processor directory")
    classifier_path = require_file(model_dir / "classifier.safetensors", "Classifier weights")

    model = BackboneLinearClassifier(
        backbone_model_name=str(backbone_path),
        num_classes=int(model_config["num_classes"]),
        freeze_backbone=True,
        local_files_only=True,
    )
    classifier_state = load_file(classifier_path)
    model.classifier.load_state_dict(classifier_state)
    model.to(device)
    model.eval()

    processor = AutoImageProcessor.from_pretrained(
        processor_path,
        local_files_only=True,
    )

    return LoadedClassifier(
        model=model,
        processor=processor,
        model_config=model_config,
        labels=labels,
        id_to_label=build_id_to_label(labels),
    )
