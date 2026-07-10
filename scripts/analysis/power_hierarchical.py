"""Honest power / minimal-detectable-effect for the paired concat-image AUROC delta.

The earlier MDE (0.010) used the patient-bootstrap SE of the *seed-averaged* delta and
therefore ignored the dominant across-seed (training-run) variance. This recomputes the
uncertainty of the delta a *single deployed model* would show, by a mixed/hierarchical
bootstrap that resamples both the training seed and the patients, and reports the honest
MDE at 80% power. Uses committed per-seed predictions only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import pandas as pd

KEYS = ["subject_id", "study_id", "dicom_id"]
SEEDS = [42, 123, 456, 789, 1000]
MS = Path("artifacts/models/multiseed")
OUT = Path("artifacts/evaluation/equivalence/power_hierarchical.json")
MARGIN = 0.05
B = 5000
RNG = np.random.default_rng(20260712)


def load(arch, seed):
    return pd.read_csv(MS / f"{arch}_seed{seed}" / "test_predictions.csv")[KEYS + ["target", "pred_prob"]]


def main():
    # align image and concat per seed on shared rows
    per_seed = {}
    for s in SEEDS:
        im = load("image", s).rename(columns={"pred_prob": "img"})
        co = load("concat", s)[KEYS + ["pred_prob"]].rename(columns={"pred_prob": "con"})
        m = im.merge(co, on=KEYS)
        per_seed[s] = m
    base = per_seed[SEEDS[0]]
    subj = base["subject_id"].to_numpy()
    y = base["target"].to_numpy()
    uniq = np.unique(subj)
    groups = [np.where(subj == u)[0] for u in uniq]

    # observed per-seed deltas
    per_seed_delta = []
    for s in SEEDS:
        m = per_seed[s]
        per_seed_delta.append(roc_auc_score(m.target, m.con) - roc_auc_score(m.target, m.img))
    per_seed_delta = np.array(per_seed_delta)

    # mixed bootstrap: draw a seed, resample patients, compute that seed's delta
    imgs = {s: per_seed[s]["img"].to_numpy() for s in SEEDS}
    cons = {s: per_seed[s]["con"].to_numpy() for s in SEEDS}
    deltas = np.empty(B)
    for b in range(B):
        s = SEEDS[RNG.integers(0, len(SEEDS))]
        ix = np.concatenate([groups[i] for i in RNG.integers(0, len(uniq), len(uniq))])
        yb = y[ix]
        if yb.min() == yb.max():
            deltas[b] = np.nan
            continue
        deltas[b] = roc_auc_score(yb, cons[s][ix]) - roc_auc_score(yb, imgs[s][ix])
    deltas = deltas[~np.isnan(deltas)]

    se = deltas.std(ddof=1)
    mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * se
    ci95 = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
    ci90 = [float(np.percentile(deltas, 5)), float(np.percentile(deltas, 95))]
    obs = float(per_seed_delta.mean())
    # TOST via normal approx on the mixed SE
    p_lo = 1 - stats.norm.cdf((obs + MARGIN) / se)
    p_hi = stats.norm.cdf((obs - MARGIN) / se)
    res = {
        "per_seed_delta": [round(float(x), 4) for x in per_seed_delta],
        "per_seed_delta_mean": round(obs, 4),
        "per_seed_delta_sd": round(float(per_seed_delta.std(ddof=1)), 4),
        "mixed_bootstrap_se": round(float(se), 4),
        "mde_80pct_power": round(float(mde), 4),
        "mixed_ci95": [round(x, 4) for x in ci95],
        "mixed_ci90": [round(x, 4) for x in ci90],
        "tost_p": round(float(max(p_lo, p_hi)), 4),
        "equivalent_at_05": bool(max(p_lo, p_hi) < 0.05),
        "margin": MARGIN,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
