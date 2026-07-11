"""Significance battery for the pneumonia dissociation (flagship fine-tuned-ensemble setup).

For each label (radiographic CheXpert, clinical ICD):
  - absolute AUROC for image / triage / fusion with patient-clustered bootstrap 95% CIs
  - DeLong paired test, image vs fusion (the standard analytic paired-AUROC test)
  - permutation test of the triage contribution: shuffle the triage prediction across test
    patients, recombine with the fixed image score through the fitted meta-learner, and
    rebuild the null distribution of the fusion-minus-image delta
  - Benjamini-Hochberg FDR across the two delta tests

Primary inference remains the patient-clustered bootstrap (DeLong assumes independent rows,
which slightly understates variance when a patient contributes several studies).

Output: artifacts/evaluation/clinical_label/significance.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.training.train_multimodal_pneumonia import build_tabular_preprocessor, prepare_tabular_df

warnings.filterwarnings("ignore")
KEYS = ["subject_id", "study_id", "dicom_id"]
SEEDS = [42, 123, 456, 789, 1000]
PHYS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity",
        "temperature_missing", "heartrate_missing", "resprate_missing", "o2sat_missing",
        "sbp_missing", "dbp_missing", "pain_missing", "acuity_missing"]
DIAG = "D:/mimic_iv_ed/diagnosis.csv.gz"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = Path("artifacts/evaluation/clinical_label/significance.json")
B = 2000
NPERM = 5000
RNG = np.random.default_rng(20260713)


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


# ---- DeLong (fast midrank implementation, Sun & Xu 2014) ----
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong_test(y, s1, s2):
    pos = y == 1
    scores = np.vstack([s1, s2])
    scores = np.hstack([scores[:, pos], scores[:, ~pos]])
    m = int(pos.sum())
    n = scores.shape[1] - m
    k = 2
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(scores[r, :m])
        ty[r] = _midrank(scores[r, m:])
        tz[r] = _midrank(scores[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    L = np.array([1.0, -1.0])
    var = L @ cov @ L
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def clustered_ci(y, s, subj):
    u = np.unique(subj)
    g = [np.where(subj == x)[0] for x in u]
    obs = roc_auc_score(y, s)
    vals = []
    for _ in range(B):
        ix = np.concatenate([g[i] for i in RNG.integers(0, len(u), len(u))])
        yb = y[ix]
        if len(np.unique(yb)) > 1:
            vals.append(roc_auc_score(yb, s[ix]))
    return round(float(obs), 3), [round(float(np.percentile(vals, 2.5)), 3),
                                  round(float(np.percentile(vals, 97.5)), 3)]


def perm_triage(y, img_l, tt, meta):
    """Permute triage predictions across patients; null distribution of fusion-image delta."""
    obs = roc_auc_score(y, meta.predict_proba(np.column_stack([img_l, logit(tt)]))[:, 1]) - roc_auc_score(y, img_l_prob(img_l))
    null = np.empty(NPERM)
    base_img = img_l_prob(img_l)
    img_auc = roc_auc_score(y, base_img)
    for i in range(NPERM):
        ttp = RNG.permutation(tt)
        fus = meta.predict_proba(np.column_stack([img_l, logit(ttp)]))[:, 1]
        null[i] = roc_auc_score(y, fus) - img_auc
    return float(obs), float((null >= obs).mean() if obs >= 0 else (null <= obs).mean())


def img_l_prob(img_logit):
    return 1 / (1 + np.exp(-img_logit))


def ens(split):
    dfs = [pd.read_csv(f"artifacts/models/multiseed/image_seed{s}/{split}_predictions.csv")
           [KEYS + ["pred_prob"]].rename(columns={"pred_prob": f"p{s}"}) for s in SEEDS]
    m = dfs[0]
    for x in dfs[1:]:
        m = m.merge(x, on=KEYS)
    m["img"] = m[[f"p{s}" for s in SEEDS]].mean(axis=1)
    return m[KEYS + ["img"]]


def icd_label(table):
    d = pd.read_csv(DIAG, usecols=["stay_id", "icd_code", "icd_version"], dtype={"icd_code": str})

    def is_pneu(code, ver):
        c = str(code).strip().upper().replace(".", "")
        if ver == 10:
            return c[:3] in {"J12", "J13", "J14", "J15", "J16", "J17", "J18"} or c[:4] == "J690"
        return c[:3] in {"480", "481", "482", "483", "484", "485", "486"} or c[:4] == "5070"

    d["pneu"] = [is_pneu(c, v) for c, v in zip(d.icd_code, d.icd_version)]
    table["icd"] = table.stay_id.map(d.groupby("stay_id").pneu.max()).fillna(False).astype(int)
    return table


def bh(pvals):
    p = np.array(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        r = m - rank
        prev = min(prev, p[idx] * m / r)
        adj[idx] = prev
    return [round(float(x), 4) for x in adj]


def main():
    d = pd.read_parquet(TABLE)
    d = icd_label(d)
    pre = build_tabular_preprocessor(PHYS, [])
    tr = d[d.temporal_split == "train"]
    pre.fit(prepare_tabular_df(tr, PHYS, []))
    va = d[d.temporal_split == "validate"].merge(ens("val"), on=KEYS)
    te = d[d.temporal_split == "test"].merge(ens("test"), on=KEYS)
    Xtr = pre.transform(prepare_tabular_df(tr, PHYS, []))
    Xva = pre.transform(prepare_tabular_df(va, PHYS, []))
    Xte = pre.transform(prepare_tabular_df(te, PHYS, []))
    subj = te.subject_id.to_numpy()

    res = {"labels": {}}
    delta_ps = []
    for lab, name in [("target", "radiographic_chexpert"), ("icd", "clinical_icd")]:
        tri = LogisticRegression(max_iter=2000).fit(Xtr, tr[lab])
        tv, tt = tri.predict_proba(Xva)[:, 1], tri.predict_proba(Xte)[:, 1]
        yt = te[lab].to_numpy()
        img_logit = logit(te.img.to_numpy())
        meta = LogisticRegression(max_iter=1000).fit(
            np.column_stack([logit(va.img), logit(tv)]), va[lab])
        fus = meta.predict_proba(np.column_stack([img_logit, logit(tt)]))[:, 1]
        img_auc, img_ci = clustered_ci(yt, te.img.to_numpy(), subj)
        tri_auc, tri_ci = clustered_ci(yt, tt, subj)
        fus_auc, fus_ci = clustered_ci(yt, fus, subj)
        _, _, z, p_delong = delong_test(yt, fus, te.img.to_numpy())
        obs, p_perm = perm_triage(yt, img_logit, tt, meta)
        delta_ps.append(p_perm)
        # ambiguous subset: image score in the middle tertile (model unsure)
        lo, hi = np.percentile(te.img, [33.3, 66.7])
        amb = (te.img.to_numpy() >= lo) & (te.img.to_numpy() <= hi)
        amb_gain = (roc_auc_score(yt[amb], fus[amb]) - roc_auc_score(yt[amb], te.img.to_numpy()[amb])
                    if len(np.unique(yt[amb])) > 1 else None)
        res["labels"][name] = {
            "image_auroc": img_auc, "image_ci95": img_ci,
            "triage_auroc": tri_auc, "triage_ci95": tri_ci,
            "fusion_auroc": fus_auc, "fusion_ci95": fus_ci,
            "fusion_minus_image": round(float(obs), 4),
            "delong_z": round(z, 3), "delong_p": round(p_delong, 4),
            "permutation_p": round(float(p_perm), 4),
            "ambiguous_band_gain": round(float(amb_gain), 4) if amb_gain is not None else None,
        }
    adj = bh(delta_ps)
    for (name, _), a in zip([("radiographic_chexpert", 0), ("clinical_icd", 0)], adj):
        res["labels"][name]["permutation_p_fdr"] = a
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    for name, r in res["labels"].items():
        print(f"{name}: fusion {r['fusion_auroc']} {r['fusion_ci95']} vs image {r['image_auroc']} "
              f"{r['image_ci95']}; delta {r['fusion_minus_image']:+.4f}; "
              f"DeLong p={r['delong_p']}, perm p={r['permutation_p']} (FDR {r['permutation_p_fdr']}); "
              f"ambiguous-band gain {r['ambiguous_band_gain']}")


if __name__ == "__main__":
    main()
