"""Shared helpers for post-hoc analyses over multi-seed predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MULTISEED_ROOT = Path("artifacts/models/multiseed")
ALIGN_KEYS = ["subject_id", "study_id", "dicom_id"]


def pred_path(arch: str, seed: int) -> Path:
    return MULTISEED_ROOT / f"{arch}_seed{seed}" / "test_predictions.csv"


def load_pred(arch: str, seed: int) -> pd.DataFrame | None:
    p = pred_path(arch, seed)
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    df["target"] = pd.to_numeric(df["target"]).astype(int)
    df["pred_prob"] = pd.to_numeric(df["pred_prob"]).astype(float)
    return df


def available_seeds(arch: str, seeds: list[int]) -> list[int]:
    return [s for s in seeds if pred_path(arch, s).is_file()]


def summarize(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def ece_uniform(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    """Expected Calibration Error, uniform-width bins (Guo et al. 2017)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        c = int(mask.sum())
        if c == 0:
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (c / n) * abs(acc - conf)
    return float(ece)


def ece_quantile(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> float:
    """Expected Calibration Error, equal-frequency (quantile) bins."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    # rank-based equal-frequency assignment, robust to ties
    order = np.argsort(y_prob, kind="mergesort")
    bin_of = np.empty(n, dtype=int)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    for b in range(n_bins):
        bin_of[order[edges[b]:edges[b + 1]]] = b
    ece = 0.0
    for b in range(n_bins):
        mask = bin_of == b
        c = int(mask.sum())
        if c == 0:
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (c / n) * abs(acc - conf)
    return float(ece)
