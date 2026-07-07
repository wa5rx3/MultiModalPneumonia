"""Tier-1 credibility statistics for the fusion-neutrality null.

All computed from committed per-seed test predictions (no retraining), on the
identical 1,075-study held-out test set, resampling at the PATIENT level.

Outputs (artifacts/evaluation/equivalence/):
  1. TOST equivalence test of the paired AUROC delta (fusion - image) against the
     pre-specified +/-0.05 margin, on the patient-level paired bootstrap distribution.
  2. Minimal-detectable-effect (MDE) at 80% power from the bootstrap SE of the delta.
  3. Patient-level bootstrap 95% CIs on ECE for each model (not just SD over seeds).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

KEYS = ["subject_id", "study_id", "dicom_id"]
SEEDS = [42, 123, 456, 789, 1000]
MS = Path("artifacts/models/multiseed")
OUT = Path("artifacts/evaluation/equivalence")
MARGIN = 0.05          # pre-specified non-inferiority / equivalence margin (AUROC)
B = 2000               # bootstrap replicates (matches the paper's paired bootstrap)
RNG = np.random.default_rng(20260707)


def patient_groups(subject_ids):
    """Precompute patient -> row-index arrays once (cluster-bootstrap unit)."""
    uniq = np.unique(subject_ids)
    groups = [np.where(subject_ids == s)[0] for s in uniq]
    return uniq, groups


def boot_index(uniq, groups):
    """One patient-level bootstrap resample -> row indices."""
    pick = RNG.integers(0, len(uniq), size=len(uniq))
    return np.concatenate([groups[i] for i in pick])


def ensemble(arch: str) -> pd.DataFrame:
    """Seed-averaged predictions for one architecture, aligned on shared test rows."""
    dfs = []
    for s in SEEDS:
        p = MS / f"{arch}_seed{s}" / "test_predictions.csv"
        dfs.append(pd.read_csv(p)[KEYS + ["target", "pred_prob"]]
                   .rename(columns={"pred_prob": f"p{s}"}))
    m = dfs[0]
    for d in dfs[1:]:
        m = m.merge(d.drop(columns="target"), on=KEYS)
    pcols = [c for c in m.columns if c.startswith("p")]
    m["pred_prob"] = m[pcols].mean(axis=1)
    return m[KEYS + ["target", "pred_prob"]]


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return e


def paired_auroc_delta(fusion: pd.DataFrame, image: pd.DataFrame):
    """Bootstrap distribution of AUROC(fusion) - AUROC(image), patient-clustered."""
    j = fusion.merge(image[KEYS + ["pred_prob"]], on=KEYS, suffixes=("_f", "_i"))
    subj = j["subject_id"].to_numpy()
    y = j["target"].to_numpy()
    pf = j["pred_prob_f"].to_numpy()
    pi = j["pred_prob_i"].to_numpy()
    obs = roc_auc_score(y, pf) - roc_auc_score(y, pi)
    uniq, groups = patient_groups(subj)
    deltas = np.empty(B)
    for b in range(B):
        ix = boot_index(uniq, groups)
        yb = y[ix]
        if yb.min() == yb.max():
            deltas[b] = np.nan
            continue
        deltas[b] = roc_auc_score(yb, pf[ix]) - roc_auc_score(yb, pi[ix])
    deltas = deltas[~np.isnan(deltas)]
    return obs, deltas, len(j)


def tost(obs, deltas, margin=MARGIN):
    se = deltas.std(ddof=1)
    # Two one-sided tests (normal approx with bootstrap SE)
    z_lo = (obs - (-margin)) / se          # H0: delta <= -margin
    z_hi = (obs - margin) / se             # H0: delta >= +margin
    p_lo = 1 - stats.norm.cdf(z_lo)        # evidence delta > -margin
    p_hi = stats.norm.cdf(z_hi)            # evidence delta < +margin
    p_tost = max(p_lo, p_hi)
    ci90 = (float(np.percentile(deltas, 5)), float(np.percentile(deltas, 95)))
    ci95 = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))
    z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
    mde = (z_a + z_b) * se
    return {
        "obs_delta": float(obs), "boot_se": float(se),
        "p_lower": float(p_lo), "p_upper": float(p_hi), "p_tost": float(p_tost),
        "equivalent_at_05": bool(p_tost < 0.05),
        "ci90": ci90, "ci95": ci95, "ci90_within_margin": bool(ci90[0] > -margin and ci90[1] < margin),
        "mde_80pct_power": float(mde),
    }


def ece_ci(df: pd.DataFrame):
    subj = df["subject_id"].to_numpy()
    y = df["target"].to_numpy()
    p = df["pred_prob"].to_numpy()
    obs = ece(y, p)
    uniq, groups = patient_groups(subj)
    vals = np.empty(B)
    for b in range(B):
        ix = boot_index(uniq, groups)
        vals[b] = ece(y[ix], p[ix])
    return {"ece": float(obs),
            "ci95": (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img = ensemble("image")
    con = ensemble("concat")
    lab = ensemble("labs")
    att = ensemble("attn")

    res = {"margin": MARGIN, "bootstrap_replicates": B, "resample_unit": "patient (subject_id)"}

    for name, fus in [("concat_minus_image", con), ("labs_minus_image", lab)]:
        obs, deltas, n = paired_auroc_delta(fus, img)
        res[name] = {"n_test": n, **tost(obs, deltas)}

    res["ece_bootstrap_ci"] = {name: ece_ci(df)
                               for name, df in [("image", img), ("concat", con),
                                                ("attn", att), ("labs", lab)]}

    json.dump(res, open(OUT / "equivalence_power.json", "w"), indent=2)

    d = res["concat_minus_image"]
    print("=== TOST: concat - image AUROC ===")
    print(f"  observed delta {d['obs_delta']:+.4f}, boot SE {d['boot_se']:.4f}")
    print(f"  90% CI [{d['ci90'][0]:+.4f}, {d['ci90'][1]:+.4f}]  within +/-{MARGIN}: {d['ci90_within_margin']}")
    print(f"  TOST p = {d['p_tost']:.2e}  -> equivalent at 0.05: {d['equivalent_at_05']}")
    print(f"  MDE @80% power = {d['mde_80pct_power']:.4f} AUROC")
    dl = res["labs_minus_image"]
    print("=== TOST: labs - image AUROC ===")
    print(f"  observed delta {dl['obs_delta']:+.4f}  90% CI [{dl['ci90'][0]:+.4f}, {dl['ci90'][1]:+.4f}]  TOST p = {dl['p_tost']:.2e}")
    print("=== ECE 95% bootstrap CIs ===")
    for k, v in res["ece_bootstrap_ci"].items():
        print(f"  {k:7s} ECE {v['ece']:.4f}  95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
