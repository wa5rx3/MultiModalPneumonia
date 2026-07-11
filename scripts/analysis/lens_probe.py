"""GO/NO-GO probe: do clinicians over-utilize radiographic cues relative to their validity?

Brunswik lens-model logic on the pneumonia diagnostic judgment. Two cue classes:
  perceptual   = 14 ML-derived radiographic finding scores
  physiological = 8 triage vitals + 8 missingness flags
Two targets:
  judgment  = the ED clinician's coded pneumonia diagnosis (cue UTILIZATION)
  criterion = an outcome-anchored reference standard (cue ECOLOGICAL VALIDITY):
              hospital discharge pneumonia diagnosis, and respiratory culture confirmation.

For each target we fit logistic models on train and, on held-out test, measure the UNIQUE
contribution of each cue class as the AUROC drop when that class is removed. The perceptual
SHARE = perc_contrib / (perc_contrib + phys_contrib). The ANCHORING INDEX =
perceptual_share(judgment) - perceptual_share(criterion): positive means clinicians lean on the
image more than the outcome warrants (and under-use physiology). Patient-clustered bootstrap CI.

A coefficient-based share (sum of |standardized logistic weights| per class) is reported as a
robustness check. GO if the anchoring index CI excludes zero for >=2 criteria.

Output: artifacts/evaluation/lens/lens_probe.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.fusion_ladder_expanded import build, KEYS, VITALS, FLAGS

warnings.filterwarnings("ignore")
COHORT = "artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet"
TRIAGE = "artifacts/manifests/cxr_ed_triage_features.parquet"
SCOREDIR = "artifacts/evaluation/multilabel_scores"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
            "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
PERC = FINDINGS
PHYS = VITALS + FLAGS
OUT = Path("artifacts/evaluation/lens/lens_probe.json")
B = 2000
RNG = np.random.default_rng(20260801)


def findings(split):
    return pd.read_csv(f"{SCOREDIR}/expanded_{split}_finding_scores.csv")


def fit_models(tr, cols, ytr):
    """Return fitted logistic on the given columns (median-impute + standardize)."""
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=3000, C=1.0))
    pipe.fit(tr[cols], ytr)
    return pipe


def perceptual_share(tr, te, ytr_col, tr_mask, te_mask):
    """AUROC-drop unique-contribution share of perceptual cues for a target, on test."""
    ytr = tr.loc[tr_mask, ytr_col].astype(int)
    yte = te.loc[te_mask, ytr_col].astype(int).to_numpy()
    trm, tem = tr[tr_mask], te[te_mask]
    full = fit_models(trm, PERC + PHYS, ytr)
    perc = fit_models(trm, PERC, ytr)
    phys = fit_models(trm, PHYS, ytr)
    pf = full.predict_proba(tem[PERC + PHYS])[:, 1]
    pp = perc.predict_proba(tem[PERC])[:, 1]
    ph = phys.predict_proba(tem[PHYS])[:, 1]
    return yte, pf, pp, ph, full, perc, phys, tem


def shares_from_scores(y, pf, pp, ph):
    a_full = roc_auc_score(y, pf)
    perc_contrib = max(a_full - roc_auc_score(y, ph), 0.0)   # drop perceptual -> physio-only
    phys_contrib = max(a_full - roc_auc_score(y, pp), 0.0)   # drop physio -> perc-only
    denom = perc_contrib + phys_contrib
    return (perc_contrib / denom) if denom > 0 else np.nan, perc_contrib, phys_contrib


def coef_share(model):
    """Sum of |standardized coef| for perceptual vs physiological blocks (full model)."""
    coefs = model.named_steps["logisticregression"].coef_[0]
    n_perc = len(PERC)
    perc = np.abs(coefs[:n_perc]).sum()
    phys = np.abs(coefs[n_perc:]).sum()
    return perc / (perc + phys)


def main():
    base = pd.read_parquet(COHORT)[KEYS + ["stay_id", "temporal_split"]]
    tf = pd.read_parquet(TRIAGE)[KEYS + VITALS + FLAGS]
    d = base.merge(tf, on=KEYS, how="left")
    d = build(d)  # adds radiographic, ed_diagnosis, discharge_diagnosis, admission, icu_transfer, culture_confirmed, mortality
    tr = d[d.temporal_split == "train"].merge(findings("train"), on=KEYS)
    te = d[d.temporal_split == "test"].merge(findings("test"), on=KEYS)

    JUDG = "ed_diagnosis"  # the ED clinician's coded pneumonia judgment
    criteria = {
        "discharge_diagnosis": (tr.hadm_id.notna(), te.hadm_id.notna()),   # admitted subset
        "culture_confirmed": (tr.hadm_id.notna(), te.hadm_id.notna()),     # admitted subset (culture 0/1)
    }

    res = {}
    for crit, (trmask, temask) in criteria.items():
        # perceptual share of the JUDGMENT and of the CRITERION on the same test cases
        yj, jf, jp, jh, jfull, _, _, tej = perceptual_share(tr, te, JUDG, trmask, temask)
        yc, cf, cp, ch, cfull, _, _, tec = perceptual_share(tr, te, crit, trmask, temask)
        sj, jpc, jph = shares_from_scores(yj, jf, jp, jh)
        sc, cpc, cph = shares_from_scores(yc, cf, cp, ch)
        idx = sj - sc
        # patient-clustered bootstrap of the anchoring index (fixed models, resample test)
        subj = tej.subject_id.to_numpy(); u = np.unique(subj); g = [np.where(subj == x)[0] for x in u]
        boots = []
        for _ in range(B):
            ix = np.concatenate([g[i] for i in RNG.integers(0, len(u), len(u))])
            if len(np.unique(yj[ix])) > 1 and len(np.unique(yc[ix])) > 1:
                s1, _, _ = shares_from_scores(yj[ix], jf[ix], jp[ix], jh[ix])
                s2, _, _ = shares_from_scores(yc[ix], cf[ix], cp[ix], ch[ix])
                if not (np.isnan(s1) or np.isnan(s2)):
                    boots.append(s1 - s2)
        boots = np.array(boots)
        res[crit] = {
            "n_test": int(temask.sum()), "judgment_pos": int(yj.sum()), "criterion_pos": int(yc.sum()),
            "perceptual_share_judgment": round(float(sj), 3),
            "perceptual_share_criterion": round(float(sc), 3),
            "anchoring_index": round(float(idx), 4),
            "anchoring_ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "p_le0": round(float((boots <= 0).mean()), 4),
            "coef_share_judgment": round(float(coef_share(jfull)), 3),
            "coef_share_criterion": round(float(coef_share(cfull)), 3),
            "judgment_perc_contrib": round(float(jpc), 4), "judgment_phys_contrib": round(float(jph), 4),
            "criterion_perc_contrib": round(float(cpc), 4), "criterion_phys_contrib": round(float(cph), 4),
        }
        print(f"[{crit}] perc-share judgment {sj:.3f} vs criterion {sc:.3f} | "
              f"anchoring {idx:+.3f} CI {res[crit]['anchoring_ci95']} p={res[crit]['p_le0']} "
              f"| coef-share J {res[crit]['coef_share_judgment']} C {res[crit]['coef_share_criterion']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
