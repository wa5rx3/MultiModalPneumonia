"""The fusion ladder with a FAIR image baseline: the full 14-finding radiographic profile.

Addresses the concern that a single pneumonia probability under-represents the image for
outcome targets. Here the image baseline for each rung is a logistic readout of the 14
multilabel CheXpert finding scores (pneumonia, cardiomegaly, edema, effusion, support devices,
...), fit per target on train. Triage is the physiology-only vitals model. Late fusion combines
the two via a validation-fit meta-learner. The question at each rung becomes: does triage
physiology add discrimination beyond a full radiographic finding profile?

Runs on the expanded ED cohort (outcomes powered). Patient-clustered bootstrap CIs.

Output: artifacts/evaluation/clinical_label/fusion_ladder_fair.json
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

from scripts.analysis.fusion_ladder_expanded import build, KEYS, VITALS, FLAGS, LADDER

warnings.filterwarnings("ignore")
COHORT = "artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet"
TRIAGE = "artifacts/manifests/cxr_ed_triage_features.parquet"
SCOREDIR = "artifacts/evaluation/multilabel_scores"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
            "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
OUT = Path("artifacts/evaluation/clinical_label/fusion_ladder_fair.json")
B = 2000
RNG = np.random.default_rng(20260725)


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def findings(split):
    return pd.read_csv(f"{SCOREDIR}/expanded_{split}_finding_scores.csv")


def main():
    d = pd.read_parquet(COHORT)[KEYS + ["stay_id", "temporal_split"]]
    tf = pd.read_parquet(TRIAGE)[KEYS + VITALS + FLAGS]
    d = d.merge(tf, on=KEYS, how="left")
    d = build(d)
    ftr, fva, fte = findings("train"), findings("val"), findings("test")
    tr = d[d.temporal_split == "train"].merge(ftr, on=KEYS)
    va = d[d.temporal_split == "validate"].merge(fva, on=KEYS)
    te = d[d.temporal_split == "test"].merge(fte, on=KEYS)

    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(tr[VITALS])
    def vit(x):
        return np.hstack([prep.transform(x[VITALS]), x[FLAGS].fillna(0).to_numpy()])

    res = {}
    for col in LADDER:
        trm, vam, tem = tr[tr[col].notna()], va[va[col].notna()], te[te[col].notna()]
        if len(tem) == 0:
            continue
        ytr = trm[col].astype(int); yv = vam[col].astype(int).to_numpy(); yt = tem[col].astype(int).to_numpy()
        if len(np.unique(yt)) < 2 or yt.sum() < 10:
            res[col] = {"test_pos": int(yt.sum()), "note": "too few positives"}
            continue
        # image baseline: logistic readout of the 14 findings, per target
        imgLR = LogisticRegression(max_iter=2000, C=1.0).fit(trm[FINDINGS], ytr)
        iv = imgLR.predict_proba(vam[FINDINGS])[:, 1]
        it = imgLR.predict_proba(tem[FINDINGS])[:, 1]
        # triage
        triLR = LogisticRegression(max_iter=3000).fit(vit(trm), ytr)
        tv = triLR.predict_proba(vit(vam))[:, 1]
        tt = triLR.predict_proba(vit(tem))[:, 1]
        meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(iv), logit(tv)]), yv)
        fus = meta.predict_proba(np.column_stack([logit(it), logit(tt)]))[:, 1]
        sj = tem.subject_id.to_numpy(); uu = np.unique(sj); gg = [np.where(sj == x)[0] for x in uu]
        obs = roc_auc_score(yt, fus) - roc_auc_score(yt, it)
        draws = np.full(B, np.nan)
        for b in range(B):
            ix = np.concatenate([gg[i] for i in RNG.integers(0, len(uu), len(uu))])
            if len(np.unique(yt[ix])) > 1:
                draws[b] = roc_auc_score(yt[ix], fus[ix]) - roc_auc_score(yt[ix], it[ix])
        dd = draws[~np.isnan(draws)]
        res[col] = {"test_pos": int(yt.sum()), "test_n": int(len(yt)), "prevalence": round(float(yt.mean()), 3),
                    "image_auroc": round(float(roc_auc_score(yt, it)), 3),
                    "triage_auroc": round(float(roc_auc_score(yt, tt)), 3),
                    "fusion_auroc": round(float(roc_auc_score(yt, fus)), 3),
                    "fusion_minus_image": round(float(obs), 4),
                    "ci95": [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)],
                    "p_le0": round(float((dd <= 0).mean()), 4)}
        e = res[col]
        print(f"{col:20s} n+={yt.sum():5d}/{len(yt):5d} prev={yt.mean():.3f} | img {e['image_auroc']} "
              f"tri {e['triage_auroc']} fus {e['fusion_auroc']} | delta {e['fusion_minus_image']:+.4f} {e['ci95']} p={e['p_le0']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
