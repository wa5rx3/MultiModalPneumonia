"""Regenerate thesis-style figures on the CURRENT rebuilt cohort, consistently.

Uses seed-averaged (5-seed ensemble) predictions for the deep models and the
canonical clinical baselines, all aligned to the same 1,075-study test set:
  ROC curves, precision-recall curves, reliability diagrams, decision-curve analysis.
Everything is derived from committed test_predictions.csv files (no retraining).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve, average_precision_score

OUT = Path("manuscript/figures")
KEYS = ["subject_id", "study_id", "dicom_id"]
SEEDS = [42, 123, 456, 789, 1000]
MS = "artifacts/models/multiseed"
CANON = "artifacts/models"

DEEP = {
    "Image-only": "image",
    "MM concat (triage)": "concat",
    "MM attention (triage)": "attn",
    "MM concat (+labs)": "labs",
}
CLINICAL = {
    "Logistic reg. (triage)": f"{CANON}/clinical_baseline_u_ignore_temporal_strong_v2/test_predictions.csv",
    "XGBoost (triage)": f"{CANON}/clinical_xgb_u_ignore_temporal_strong_v2/test_predictions.csv",
}
COLORS = {
    "Image-only": "#4C72B0", "MM concat (triage)": "#DD8452", "MM attention (triage)": "#55A868",
    "MM concat (+labs)": "#C44E52", "Logistic reg. (triage)": "#8172B3", "XGBoost (triage)": "#937860",
}


def load_ensemble(arch: str) -> pd.DataFrame:
    dfs = []
    for s in SEEDS:
        p = Path(MS) / f"{arch}_seed{s}" / "test_predictions.csv"
        if p.is_file():
            dfs.append(pd.read_csv(p)[KEYS + ["target", "pred_prob"]].rename(columns={"pred_prob": f"p{s}"}))
    m = dfs[0]
    for d in dfs[1:]:
        m = m.merge(d.drop(columns="target"), on=KEYS)
    pcols = [c for c in m.columns if c.startswith("p")]
    m["pred_prob"] = m[pcols].mean(axis=1)
    return m[KEYS + ["target", "pred_prob"]]


def build_all() -> dict[str, pd.DataFrame]:
    preds = {name: load_ensemble(arch) for name, arch in DEEP.items()}
    ref = preds["Image-only"][KEYS]
    for name, path in CLINICAL.items():
        d = pd.read_csv(path)[KEYS + ["target", "pred_prob"]]
        preds[name] = ref.merge(d, on=KEYS, how="inner")
    return preds


def fig_roc(preds):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for name, d in preds.items():
        y, p = d["target"].to_numpy(), d["pred_prob"].to_numpy()
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=COLORS[name], lw=1.8, label=f"{name} (AUROC {roc_auc_score(y,p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test $n{=}1{,}075$)"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig7_roc.png", dpi=200); plt.close(fig)


def fig_pr(preds):
    prev = preds["Image-only"]["target"].mean()
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for name, d in preds.items():
        y, p = d["target"].to_numpy(), d["pred_prob"].to_numpy()
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=COLORS[name], lw=1.8, label=f"{name} (AUPRC {average_precision_score(y,p):.3f})")
    ax.axhline(prev, color="k", ls="--", lw=1, alpha=0.5, label=f"Prevalence {prev:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0.0, 1.0)
    ax.set_title("Precision-recall curves")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig8_pr.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_reliability(preds):
    fig, ax = plt.subplots(figsize=(6, 5.6))
    bins = np.linspace(0, 1, 11)
    for name, d in preds.items():
        y, p = d["target"].to_numpy(), d["pred_prob"].to_numpy()
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 5:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        ax.plot(xs, ys, "o-", color=COLORS[name], ms=4, lw=1.4, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagrams (10 bins)"); ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig9_reliability.png", dpi=200); plt.close(fig)


def fig_dca(preds):
    thr = np.linspace(0.01, 0.80, 80)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    y_ref = preds["Image-only"]["target"].to_numpy()
    n = len(y_ref); prev = y_ref.mean()
    # treat-all / treat-none references
    nb_all = prev - (1 - prev) * (thr / (1 - thr))
    ax.plot(thr, nb_all, color="gray", lw=1.2, ls="--", label="Treat all")
    ax.axhline(0, color="k", lw=1, ls=":", label="Treat none")
    for name, d in preds.items():
        y, p = d["target"].to_numpy(), d["pred_prob"].to_numpy()
        nb = []
        for t in thr:
            pred = p >= t
            tp = np.sum((pred == 1) & (y == 1)); fp = np.sum((pred == 1) & (y == 0))
            nb.append(tp / n - (fp / n) * (t / (1 - t)))
        ax.plot(thr, nb, color=COLORS[name], lw=1.6, label=name)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_ylim(-0.05, prev + 0.03)
    ax.set_title("Decision-curve analysis"); ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig10_dca.png", dpi=200); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preds = build_all()
    print("aligned rows per model:", {k: len(v) for k, v in preds.items()})
    fig_roc(preds); fig_pr(preds); fig_reliability(preds); fig_dca(preds)
    print("wrote fig7_roc, fig8_pr, fig9_reliability, fig10_dca to", OUT)


if __name__ == "__main__":
    main()
