"""MedFuse-style late-fusion baseline (decision-level fusion of unimodal encoders).

MedFuse and related late-fusion designs combine *independently trained* unimodal
encoders rather than jointly encoding a concatenated representation. Our clinical data
are static triage (not time-series), so the faithful static analogue of that design is
a stacked meta-learner: the independently trained DenseNet image model and the
triage-only clinical model each emit a probability, and a logistic-regression combiner
is fit on the VALIDATION split and evaluated on TEST. If this different fusion design
is also discrimination-neutral versus image-only, the null is not an artefact of our
concat/attention design.

No GPU retraining: uses committed per-seed image predictions (val+test) and the
committed triage logistic-regression baseline (val+test). Outputs
artifacts/evaluation/late_fusion/late_fusion.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

KEYS = ["subject_id", "study_id", "dicom_id"]
SEEDS = [42, 123, 456, 789, 1000]
MS = Path("artifacts/models/multiseed")
TRIAGE = Path("artifacts/models/clinical_baseline_u_ignore_temporal_strong_v2")
OUT = Path("artifacts/evaluation/late_fusion")
MARGIN = 0.05
B = 2000
RNG = np.random.default_rng(20260707)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def load(path):
    return pd.read_csv(path)[KEYS + ["target", "pred_prob"]]


def build_seed(seed, tri_val, tri_te):
    """Late-fusion test probabilities for one image seed via a val-fit LR stacker."""
    img_val = load(MS / f"image_seed{seed}" / "val_predictions.csv").rename(columns={"pred_prob": "img"})
    img_te = load(MS / f"image_seed{seed}" / "test_predictions.csv").rename(columns={"pred_prob": "img"})
    v = img_val.merge(tri_val, on=KEYS, suffixes=("", "_t"))
    t = img_te.merge(tri_te, on=KEYS, suffixes=("", "_t"))
    Xv = np.column_stack([logit(v["img"]), logit(v["tri"])])
    Xt = np.column_stack([logit(t["img"]), logit(t["tri"])])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xv, v["target"].to_numpy())
    t = t.copy()
    t["late"] = clf.predict_proba(Xt)[:, 1]
    return t[KEYS + ["target", "img", "late"]], clf.coef_[0].tolist()


def patient_groups(subj):
    uniq = np.unique(subj)
    return uniq, [np.where(subj == s)[0] for s in uniq]


def tost_delta(df, col_fus, col_img):
    subj = df["subject_id"].to_numpy(); y = df["target"].to_numpy()
    pf = df[col_fus].to_numpy(); pi = df[col_img].to_numpy()
    obs = roc_auc_score(y, pf) - roc_auc_score(y, pi)
    uniq, groups = patient_groups(subj)
    d = np.empty(B)
    for b in range(B):
        ix = np.concatenate([groups[i] for i in RNG.integers(0, len(uniq), len(uniq))])
        yb = y[ix]
        d[b] = np.nan if yb.min() == yb.max() else roc_auc_score(yb, pf[ix]) - roc_auc_score(yb, pi[ix])
    d = d[~np.isnan(d)]
    se = d.std(ddof=1)
    p_lo = 1 - stats.norm.cdf((obs + MARGIN) / se)
    p_hi = stats.norm.cdf((obs - MARGIN) / se)
    return {"obs_delta": float(obs), "boot_se": float(se), "p_tost": float(max(p_lo, p_hi)),
            "equivalent_at_05": bool(max(p_lo, p_hi) < 0.05),
            "ci90": [float(np.percentile(d, 5)), float(np.percentile(d, 95))]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tri_val = load(TRIAGE / "val_predictions.csv").rename(columns={"pred_prob": "tri"})
    tri_te = load(TRIAGE / "test_predictions.csv").rename(columns={"pred_prob": "tri"})

    per_seed, coefs = [], []
    ens = None
    for s in SEEDS:
        t, coef = build_seed(s, tri_val, tri_te)
        coefs.append(coef)
        y = t["target"].to_numpy()
        per_seed.append({"seed": s,
                         "late_auroc": roc_auc_score(y, t["late"]),
                         "img_auroc": roc_auc_score(y, t["img"]),
                         "late_auprc": average_precision_score(y, t["late"]),
                         "late_ece": ece(y, t["late"].to_numpy()),
                         "delta_auroc": roc_auc_score(y, t["late"]) - roc_auc_score(y, t["img"])})
        ens = t[KEYS + ["target", "img"]].copy() if ens is None else ens
        ens[f"late{s}"] = t.set_index(KEYS).loc[ens.set_index(KEYS).index, "late"].to_numpy()

    late_cols = [f"late{s}" for s in SEEDS]
    ens["late"] = ens[late_cols].mean(axis=1)

    ps = pd.DataFrame(per_seed)
    res = {
        "design": "decision-level late fusion (LR stacker on [logit img, logit triage], fit on validation)",
        "unimodal_clinical": "triage logistic-regression baseline",
        "per_seed": per_seed,
        "late_auroc_mean_sd": [float(ps.late_auroc.mean()), float(ps.late_auroc.std(ddof=1))],
        "img_auroc_mean_sd": [float(ps.img_auroc.mean()), float(ps.img_auroc.std(ddof=1))],
        "delta_auroc_mean_sd": [float(ps.delta_auroc.mean()), float(ps.delta_auroc.std(ddof=1))],
        "late_ece_mean_sd": [float(ps.late_ece.mean()), float(ps.late_ece.std(ddof=1))],
        "mean_stacker_coef_[img,triage]": np.mean(coefs, axis=0).tolist(),
        "ensemble_tost_late_minus_image": tost_delta(ens, "late", "img"),
    }
    json.dump(res, open(OUT / "late_fusion.json", "w"), indent=2)

    print("=== MedFuse-style late fusion (decision-level stacking) ===")
    print(f"  late-fusion AUROC {res['late_auroc_mean_sd'][0]:.4f} +/- {res['late_auroc_mean_sd'][1]:.4f}")
    print(f"  image-only  AUROC {res['img_auroc_mean_sd'][0]:.4f} +/- {res['img_auroc_mean_sd'][1]:.4f}")
    print(f"  delta (late-image) {res['delta_auroc_mean_sd'][0]:+.4f} +/- {res['delta_auroc_mean_sd'][1]:.4f}")
    print(f"  late-fusion ECE {res['late_ece_mean_sd'][0]:.4f} +/- {res['late_ece_mean_sd'][1]:.4f}")
    tt = res["ensemble_tost_late_minus_image"]
    print(f"  ensemble TOST delta {tt['obs_delta']:+.4f}  90% CI [{tt['ci90'][0]:+.4f},{tt['ci90'][1]:+.4f}]  equiv@0.05: {tt['equivalent_at_05']}")
    print(f"  mean stacker weights [img, triage] = {res['mean_stacker_coef_[img,triage]']}")


if __name__ == "__main__":
    main()
