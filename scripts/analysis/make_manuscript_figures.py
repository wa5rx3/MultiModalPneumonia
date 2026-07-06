"""Generate headline manuscript figures from committed multi-seed artifacts.

Fig 1: per-model AUROC and ECE across seeds (points = seeds, bar = mean).
Fig 2: paired deltas vs image-only (mean +/- SD across seeds) for AUROC and ECE.
All inputs are the reproducible artifacts under artifacts/evaluation/multiseed/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL = Path("artifacts/evaluation/multiseed")
OUT = Path("manuscript/figures")
LABELS = {"image": "Image-only", "concat": "MM concat\n(triage)", "attn": "MM attn\n(triage)",
          "labs": "MM concat\n(+labs)", "labflags": "MM\n(lab flags only)"}
ORDER = ["image", "concat", "attn", "labs", "labflags"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per = pd.read_csv(EVAL / "per_seed_metrics.csv")
    summ = json.load(open(EVAL / "multiseed_summary.json"))
    archs = [a for a in ORDER if a in per["arch"].unique()]

    # ---- Figure 1: AUROC and ECE across seeds ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in [(axes[0], "auroc", "AUROC (higher better)"),
                              (axes[1], "ece", "ECE (lower better)")]:
        for i, a in enumerate(archs):
            vals = per[per["arch"] == a][metric].values
            ax.scatter(np.full_like(vals, i, dtype=float), vals, color="#4C72B0", alpha=0.7, zorder=3, s=28)
            ax.hlines(vals.mean(), i - 0.28, i + 0.28, color="#C44E52", lw=2.5, zorder=4)
        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels([LABELS[a] for a in archs], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].axhline(per[per.arch == "image"]["auroc"].mean(), ls="--", color="gray", alpha=0.6, lw=1)
    fig.suptitle("Multi-seed performance (5 seeds; red bar = mean)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_multiseed_metrics.png", dpi=200)
    plt.close(fig)

    # ---- Figure 2: paired deltas vs image ----
    pv = summ.get("paired_vs_image", {})
    fusion = [a for a in ["concat", "attn", "labs", "labflags"] if a in pv]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, dk, title in [(axes[0], "delta_auroc", "ΔAUROC vs image-only"),
                          (axes[1], "delta_ece", "ΔECE vs image-only")]:
        means = [pv[a][dk]["mean"] for a in fusion]
        sds = [pv[a][dk]["std"] for a in fusion]
        y = np.arange(len(fusion))
        ax.errorbar(means, y, xerr=sds, fmt="o", color="#4C72B0", capsize=4, ms=7, zorder=3)
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[a].replace("\n", " ") for a in fusion], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
    fig.suptitle("Paired fusion − image effect across seeds (mean ± SD)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_paired_deltas.png", dpi=200)
    plt.close(fig)

    print(f"Wrote {OUT}/fig1_multiseed_metrics.png and fig2_paired_deltas.png")


if __name__ == "__main__":
    main()
