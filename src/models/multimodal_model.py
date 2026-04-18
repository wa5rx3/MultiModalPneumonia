from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseNetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = models.densenet121(weights="IMAGENET1K_V1")
        self.features = base.features
        self.out_dim = base.classifier.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ]
        )
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend(
            [
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        )

        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalPneumoniaModel(nn.Module):
    def __init__(
        self,
        tabular_input_dim: int,
        tabular_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.image_backbone = DenseNetBackbone()
        self.tabular_branch = TabularMLP(
            input_dim=tabular_input_dim,
            hidden_dim=tabular_hidden_dim,
            dropout=dropout,
            use_batchnorm=True,
        )

        fusion_input_dim = self.image_backbone.out_dim + self.tabular_branch.out_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_backbone(image)
        tab_feat = self.tabular_branch(tabular)
        fused = torch.cat([image_feat, tab_feat], dim=1)
        return self.fusion_head(fused)

    def _extract_backbone_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        multimodal_filtered = {
            k.replace("image_backbone.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("image_backbone.")
        }
        if multimodal_filtered:
            return multimodal_filtered


        image_only_filtered = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("features.")
        }
        if image_only_filtered:
            return image_only_filtered


        direct_filtered = {
            k: v
            for k, v in state_dict.items()
            if k in self.image_backbone.state_dict()
        }
        return direct_filtered

    def load_image_backbone_from_checkpoint(self, checkpoint_path: str) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" not in ckpt:
            raise KeyError(
                f"Checkpoint {checkpoint_path} does not contain 'model_state_dict'."
            )

        state_dict = ckpt["model_state_dict"]
        filtered = self._extract_backbone_state_dict(state_dict)

        if not filtered:
            raise ValueError(
                f"Could not extract image backbone weights from checkpoint: {checkpoint_path}"
            )

        missing, unexpected = self.image_backbone.load_state_dict(filtered, strict=False)
        print("Loaded image backbone checkpoint.")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    def freeze_image_backbone(self) -> None:
        for p in self.image_backbone.parameters():
            p.requires_grad = False

    def unfreeze_image_backbone(self) -> None:
        for p in self.image_backbone.parameters():
            p.requires_grad = True

    def image_backbone_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.image_backbone.parameters() if p.requires_grad)

    def total_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)