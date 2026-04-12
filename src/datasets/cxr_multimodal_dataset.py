from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CXRMultimodalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tabular_array: np.ndarray,
        transform: Callable | None = None,
        image_col: str = "image_path",
        target_col: str = "target",
    ) -> None:
        if len(df) != len(tabular_array):
            raise ValueError(
                f"Row mismatch between dataframe ({len(df)}) and tabular array ({len(tabular_array)})"
            )

        required_cols = ["subject_id", "study_id", "dicom_id", image_col, target_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = df.reset_index(drop=True).copy()
        self.tabular_array = np.asarray(tabular_array, dtype=np.float32)
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
            "tabular": torch.tensor(self.tabular_array[idx], dtype=torch.float32),
            "target": torch.tensor(float(row[self.target_col]), dtype=torch.float32),
            "subject_id": int(row["subject_id"]),
            "study_id": int(row["study_id"]),
            "dicom_id": row["dicom_id"],
            "image_path": row[self.image_col],
        }