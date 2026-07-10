"""Reliability-diagram grid (fig17): calibration curves for all five model variants.

One panel per model (seed 42, single checkpoint), each showing the reliability curve
(mean predicted probability vs observed frequency, 10 uniform bins), the prediction
histogram, the perfect-calibration diagonal, and the ECE. This is a single-checkpoint
view: the manuscript's Caution 1 covers how the concat calibration edge narrows across
seeds. Style follows the thesis reliability figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("manuscript/figures/fig17_reliability_grid.png")
M = "artifacts/models"
MODELS = [
    ("Image-only (DenseNet-121)", f"{M}/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv", "#4C72B0"),
    ("Multimodal-concat", f"{M}/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv", "#DD8452"),
    ("Multimodal-attention", f"{M}/multimodal_pneumonia_attn_fusion_u_ignore_temporal_v1/test_predictions.csv", "#55A868"),
    ("Clinical XGBoost", f"{M}/clinical_xgb_u_ignore_temporal_strong_v2/test_predictions.csv", "#C44E52"),
    ("Clinical LR", f"{M}/clinical_baseline_u_ignore_temporal_strong_v2/test_predictions.csv", "#8172B3"),
]


def ece_10(y, p):
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
    e = 0.0
    for b in range(10):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def panel(ax, title, path, color):
    d = pd.read_csv(path)
    y, p = d["target"].to_numpy(), d["pred_prob"].to_numpy()
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
    xs, ys = [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 1:
            xs.append(p[m].mean()); ys.append(y[m].mean())
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.fill_between(xs, xs, ys, color=color, alpha=0.15)
    ax.plot(xs, ys, "o-", color=color, lw=1.8, ms=5, label="Model calibration")
    # prediction histogram along the bottom
    counts, edges = np.histogram(p, bins=bins)
    ax.bar(edges[:-1] + 0.05, counts / counts.max() * 0.28, width=0.09,
           color=color, alpha=0.25, align="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(title, color=color, fontsize=10, fontweight="bold")
    ax.text(0.04, 0.95, f"ECE = {ece_10(y, p):.3f}", transform=ax.transAxes,
            fontsize=9, va="top", bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.grid(alpha=0.25)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8))
    axes = axes.ravel()
    for ax, (t, path, c) in zip(axes, MODELS):
        panel(ax, t, path, c)
    axes[-1].axis("off")
    axes[-1].plot([], [], "k--", label="Perfect calibration")
    axes[-1].legend(loc="center", fontsize=11, frameon=False)
    fig.suptitle("Reliability diagrams: test set (n=1,075, 10 uniform bins, seed 42)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {OUT}")
    for t, path, _ in MODELS:
        d = pd.read_csv(path)
        print(f"  {t}: ECE {ece_10(d.target.to_numpy(), d.pred_prob.to_numpy()):.3f}")


if __name__ == "__main__":
    main()
