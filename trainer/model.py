import torch
from torch import nn
from transformers import AutoModel


class BackboneLinearClassifier(nn.Module):
    """Backbone plus a linear classification head."""

    def __init__(
        self,
        *,
        backbone_model_name: str,
        num_classes: int,
        freeze_backbone: bool,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone_frozen = True
        self.backbone.eval()

    def unfreeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True
        self.backbone_frozen = False

    def train(self, mode: bool = True) -> "BackboneLinearClassifier":
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.backbone_frozen:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values)
        if getattr(outputs, "pooler_output", None) is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state[:, 0]
        return self.classifier(features)
