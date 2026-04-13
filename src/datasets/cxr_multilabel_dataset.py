from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


CHEXPERT_LABEL_COLS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


class CXRMultilabelDataset(Dataset):
    def __init__(
        self,
        table_path: str,
        split: str,
        transform: Callable | None = None,
        image_col: str = "image_path",
        skip_missing: bool = True,
    ) -> None:
        self.df = pd.read_parquet(table_path).copy()

        if "pretrain_split" not in self.df.columns:
            raise ValueError("Expected column 'pretrain_split' not found.")

        self.df = self.df[self.df["pretrain_split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split='{split}'")

        if skip_missing and image_col in self.df.columns:
            exists_mask = self.df[image_col].map(lambda p: Path(p).exists())
            n_missing = int((~exists_mask).sum())
            if n_missing > 0:
                print(f"[CXRMultilabelDataset] Skipping {n_missing} rows with missing images (split={split})")
                self.df = self.df[exists_mask].reset_index(drop=True)

        self.transform = transform
        self.image_col = image_col

        missing_cols = [c for c in CHEXPERT_LABEL_COLS if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing label columns: {missing_cols}")

        missing_mask_cols = [f"{c}_mask" for c in CHEXPERT_LABEL_COLS if f"{c}_mask" not in self.df.columns]
        if missing_mask_cols:
            raise ValueError(f"Missing mask columns: {missing_mask_cols}")

        for col in CHEXPERT_LABEL_COLS:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            self.df[f"{col}_mask"] = self.df[f"{col}_mask"].fillna(False).astype(bool)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, image_path: str) -> Image.Image:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            return Image.open(path).convert("RGB")
        except OSError as e:
            raise OSError(f"Failed to load image: {image_path} ; error={e}")

    @staticmethod
    def _build_target_and_mask(row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        target = []
        mask = []

        for col in CHEXPERT_LABEL_COLS:
            raw_value = row[col]
            raw_mask = bool(row[f"{col}_mask"])

            if raw_mask:
                target.append(float(raw_value))
                mask.append(1.0)
            else:
                target.append(0.0)
                mask.append(0.0)

        return torch.tensor(target, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image = self._load_image(row[self.image_col])
        if self.transform is not None:
            image = self.transform(image)

        target, mask = self._build_target_and_mask(row)

        return {
            "image": image,
            "target": target,
            "mask": mask,
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "dicom_id": row["dicom_id"],
            "image_path": row[self.image_col],
        }