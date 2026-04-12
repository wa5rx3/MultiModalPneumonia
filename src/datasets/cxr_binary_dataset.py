from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CXRBinaryDataset(Dataset):
    def __init__(
        self,
        table_path: str,
        split: str,
        transform: Callable | None = None,
        image_col: str = "image_path",
        target_col: str = "target",
        split_col: str = "temporal_split",
    ) -> None:
        self.df = pd.read_parquet(table_path).copy()

        if split_col not in self.df.columns:
            raise ValueError(f"Expected split column '{split_col}' not found.")
        if target_col not in self.df.columns:
            raise ValueError(f"Expected target column '{target_col}' not found.")

        self.df = self.df[self.df[split_col] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows found for split='{split}'")

        self.transform = transform
        self.image_col = image_col
        self.target_col = target_col

        self.df[self.target_col] = pd.to_numeric(self.df[self.target_col], errors="raise").astype(int)

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

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image = self._load_image(row[self.image_col])
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "target": torch.tensor(float(row[self.target_col]), dtype=torch.float32),
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "dicom_id": row["dicom_id"],
            "image_path": row[self.image_col],
        }