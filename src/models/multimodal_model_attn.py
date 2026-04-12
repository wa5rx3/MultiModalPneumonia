from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.multimodal_model import DenseNetBackbone, TabularMLP


class MultimodalPneumoniaModelAttn(nn.Module):
    """Attention fusion: DenseNet-121 image features + tabular MLP features combined via Transformer CLS token."""

    def __init__(
        self,
        tabular_input_dim: int,
        tabular_hidden_dim: int = 128,
        d_model: int = 256,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.image_backbone = DenseNetBackbone()
        self.tabular_branch = TabularMLP(
            input_dim=tabular_input_dim,
            hidden_dim=tabular_hidden_dim,
            dropout=dropout,
            use_batchnorm=True,
        )

        self.image_proj = nn.Linear(self.image_backbone.out_dim, d_model)
        self.tabular_proj = nn.Linear(tabular_hidden_dim, d_model)

        self.type_embed = nn.Embedding(2, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_backbone(image)
        tab_feat = self.tabular_branch(tabular)

        image_tok = self.image_proj(image_feat)
        tab_tok = self.tabular_proj(tab_feat)

        B = image_tok.shape[0]
        device = image_tok.device

        image_tok = image_tok + self.type_embed(torch.zeros(B, dtype=torch.long, device=device))
        tab_tok = tab_tok + self.type_embed(torch.ones(B, dtype=torch.long, device=device))

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, image_tok.unsqueeze(1), tab_tok.unsqueeze(1)], dim=1)

        out = self.transformer(seq)
        cls_out = out[:, 0, :]

        return self.classifier(self.norm(cls_out))

    def _extract_backbone_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        multimodal_filtered = {
            k.replace("image_backbone.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("image_backbone.")
        }
        if multimodal_filtered:
            return multimodal_filtered

        image_only_filtered = {
            k: v for k, v in state_dict.items() if k.startswith("features.")
        }
        if image_only_filtered:
            return image_only_filtered

        direct_filtered = {
            k: v for k, v in state_dict.items() if k in self.image_backbone.state_dict()
        }
        return direct_filtered

    def load_image_backbone_from_checkpoint(self, checkpoint_path: str) -> None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" not in ckpt:
            raise KeyError(f"Checkpoint {checkpoint_path} does not contain 'model_state_dict'.")

        state_dict = ckpt["model_state_dict"]
        filtered = self._extract_backbone_state_dict(state_dict)

        if not filtered:
            raise ValueError(f"Could not extract image backbone weights from checkpoint: {checkpoint_path}")

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
