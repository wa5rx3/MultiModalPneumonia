"""Secondary manuscript figures: subgroup fairness, lab coverage, ECE bin sensitivity.
All inputs are committed artifacts; no data access needed."""
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
LABS_REPORT = Path("artifacts/logs/cxr_clinical_labs_training_table_u_ignore_temporal_report.json")


def fig_subgroup() -> None:
    df = pd.read_csv(EVAL / "subgroup_metrics.csv")
    au = df[df["metric"] == "auroc"]
    # pick informative, adequately-sized levels
    levels = [("sex", "Male"), ("sex", "Female"),
              ("race_group", "White"), ("race_group", "Black"), ("race_group", "Hispanic"),
              ("view", "PA"), ("view", "AP"),
              ("acuity_group", "High acuity (ESI 1-2)"), ("acuity_group", "Lower acuity (ESI 3-5)")]
    names, img_m, img_s, con_m, con_s = [], [], [], [], []
    for factor, level in levels:
        sub = au[(au["factor"] == factor) & (au["level"] == level)]
        if sub.empty:
            continue
        gi = sub[sub["arch"] == "image"]; gc = sub[sub["arch"] == "concat"]
        if gi.empty or gc.empty:
            continue
        names.append(level.replace(" (ESI 1-2)", "").replace(" (ESI 3-5)", ""))
        img_m.append(gi["mean"].iloc[0]); img_s.append(gi["std"].iloc[0])
        con_m.append(gc["mean"].iloc[0]); con_s.append(gc["std"].iloc[0])
    y = np.arange(len(names)); h = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(y + h/2, img_m, height=h, xerr=img_s, label="Image-only", color="#4C72B0", capsize=3)
    ax.barh(y - h/2, con_m, height=h, xerr=con_s, label="MM concat (triage)", color="#DD8452", capsize=3)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
    ax.set_xlim(0.5, 0.85); ax.set_xlabel("AUROC (mean ± SD across seeds)")
    ax.set_title("Subgroup discrimination (fusion does not close gaps)")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig3_subgroup_auroc.png", dpi=200); plt.close(fig)


def fig_lab_coverage() -> None:
    rep = json.load(open(LABS_REPORT))
    cov = rep["per_lab_coverage_pct_test"]
    items = sorted(cov.items(), key=lambda x: x[1])
    names = [k for k, _ in items]; vals = [v for _, v in items]
    colors = ["#C44E52" if k in ("crp", "procalcitonin") else "#55A868" for k in names]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("% of test studies with value resolved at/before t0")
    ax.set_title("Lab availability at imaging time\n(red = pneumonia-specific markers, ~0%)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig4_lab_coverage.png", dpi=200); plt.close(fig)


def fig_ece_bins() -> None:
    df = pd.read_csv(EVAL / "ece_bin_sensitivity.csv")
    schemes = df[["scheme", "n_bins"]].drop_duplicates().values.tolist()
    xlabels = [f"{s}\n{n}" for s, n in schemes]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for arch, color in [("image", "#4C72B0"), ("concat", "#DD8452")]:
        means, sds = [], []
        for s, n in schemes:
            v = df[(df.arch == arch) & (df.scheme == s) & (df.n_bins == n)]["ece"].values
            means.append(v.mean()); sds.append(v.std(ddof=1) if len(v) > 1 else 0)
        ax.errorbar(range(len(schemes)), means, yerr=sds, fmt="o-", color=color,
                    label={"image": "Image-only", "concat": "MM concat"}[arch], capsize=3)
    ax.set_xticks(range(len(schemes))); ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("ECE (mean ± SD across seeds)")
    ax.set_title("Calibration gap depends on binning scheme")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig5_ece_bin_sensitivity.png", dpi=200); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_subgroup(); fig_lab_coverage(); fig_ece_bins()
    print(f"Wrote fig3_subgroup_auroc.png, fig4_lab_coverage.png, fig5_ece_bin_sensitivity.png to {OUT}/")


if __name__ == "__main__":
    main()
