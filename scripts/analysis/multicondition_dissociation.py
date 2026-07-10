"""Multi-condition radiographic-vs-clinical fusion dissociation (pneumonia + heart failure).

For each condition, the image signal is the multilabel model's finding score (Pneumonia,
Edema), the radiographic label is the CheXpert finding, the clinical label is the ED-diagnosis
ICD code. Late fusion combines the finding score with a physiology-only triage model (meta-learner
fit on validation). Reports fusion-image AUROC deltas vs each label with patient-level bootstrap CIs,
plus the interaction test (clinical delta - radiographic delta).

Output: artifacts/evaluation/clinical_label/multicondition.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.training.train_multimodal_pneumonia import build_tabular_preprocessor, prepare_tabular_df

warnings.filterwarnings("ignore")
KEYS = ["subject_id", "study_id", "dicom_id"]
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
SCORES = "artifacts/evaluation/multilabel_scores"
CHEX = "D:/mimic_data/mimic-cxr-2.0.0-chexpert.csv.gz"
DIAG = "D:/mimic_iv_ed/diagnosis.csv.gz"
OUT = Path("artifacts/evaluation/clinical_label/multicondition.json")
PHYS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity",
        "temperature_missing", "heartrate_missing", "resprate_missing", "o2sat_missing",
        "sbp_missing", "dbp_missing", "pain_missing", "acuity_missing"]
B = 2000
RNG = np.random.default_rng(20260712)
CONDS = {  # condition: (image finding score col, CheXpert finding, icd10 prefixes, icd9 prefixes)
    "pneumonia": ("Pneumonia", "Pneumonia", ("J12", "J13", "J14", "J15", "J16", "J17", "J18"),
                  ("480", "481", "482", "483", "484", "485", "486")),
    "heart_failure": ("Edema", "Edema", ("I50",), ("428",)),
}


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def groups(subj):
    u = np.unique(subj)
    return u, [np.where(subj == x)[0] for x in u]


def boot(y, img, fus, subj):
    u, g = groups(subj)
    obs = roc_auc_score(y, fus) - roc_auc_score(y, img)
    ds = []
    for _ in range(B):
        ix = np.concatenate([g[i] for i in RNG.integers(0, len(u), len(u))])
        yb = y[ix]
        if len(np.unique(yb)) > 1:
            ds.append(roc_auc_score(yb, fus[ix]) - roc_auc_score(yb, img[ix]))
    return obs, np.array(ds)


def main():
    d = pd.read_parquet(TABLE)
    cx = pd.read_csv(CHEX).set_index("study_id")
    diag = pd.read_csv(DIAG, usecols=["stay_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    diag["c"] = diag.icd_code.str.upper().str.replace(".", "", regex=False)
    sv = pd.read_csv(f"{SCORES}/val_finding_scores.csv"); st = pd.read_csv(f"{SCORES}/test_finding_scores.csv")

    tr = d[d.temporal_split == "train"]
    va = d[d.temporal_split == "validate"].merge(sv, on=KEYS)
    te = d[d.temporal_split == "test"].merge(st, on=KEYS)
    pre = build_tabular_preprocessor(PHYS, [])
    Xtr = pre.fit_transform(prepare_tabular_df(tr, PHYS, []))
    Xva = pre.transform(prepare_tabular_df(va, PHYS, []))
    Xte = pre.transform(prepare_tabular_df(te, PHYS, []))

    def rad(df, find):
        return (cx.reindex(df.study_id.values)[find] == 1).fillna(False).astype(int).values

    def clin(df, p10, p9):
        m = ((diag.icd_version == 10) & diag.c.str.startswith(tuple(p10))) | ((diag.icd_version == 9) & diag.c.str.startswith(tuple(p9)))
        stays = set(diag.loc[m, "stay_id"])
        return df.stay_id.isin(stays).astype(int).values

    res = {}
    for name, (score, find, p10, p9) in CONDS.items():
        labels = {"radiographic": (rad(tr, find), rad(va, find), rad(te, find)),
                  "clinical": (clin(tr, p10, p9), clin(va, p10, p9), clin(te, p10, p9))}
        cond = {"image_finding": score, "clinical_test_pos": int(labels["clinical"][2].sum())}
        deltas = {}
        for lab, (ytr, yva, yte) in labels.items():
            tri = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
            tv, tt = tri.predict_proba(Xva)[:, 1], tri.predict_proba(Xte)[:, 1]
            imgv, imgt = va[score].to_numpy(), te[score].to_numpy()
            meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(imgv), logit(tv)]), yva)
            fus = meta.predict_proba(np.column_stack([logit(imgt), logit(tt)]))[:, 1]
            obs, ds = boot(yte, imgt, fus, te.subject_id.to_numpy())
            deltas[lab] = ds
            cond[lab] = {"image_auroc": round(roc_auc_score(yte, imgt), 3),
                         "triage_auroc": round(roc_auc_score(yte, tt), 3),
                         "fusion_auroc": round(roc_auc_score(yte, fus), 3),
                         "fusion_minus_image": round(float(obs), 4),
                         "ci95": [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)],
                         "p_le0": round(float((ds <= 0).mean()), 4)}
        n = min(len(deltas["clinical"]), len(deltas["radiographic"]))
        inter = deltas["clinical"][:n] - deltas["radiographic"][:n]
        cond["interaction_clinical_minus_radiographic"] = {
            "estimate": round(float(cond["clinical"]["fusion_minus_image"] - cond["radiographic"]["fusion_minus_image"]), 4),
            "ci95": [round(float(np.percentile(inter, 2.5)), 4), round(float(np.percentile(inter, 97.5)), 4)],
            "p_le0": round(float((inter <= 0).mean()), 4)}
        res[name] = cond
        print(f"{name}: radiographic delta {cond['radiographic']['fusion_minus_image']:+.3f}, "
              f"clinical delta {cond['clinical']['fusion_minus_image']:+.3f}, "
              f"interaction {cond['interaction_clinical_minus_radiographic']['estimate']:+.3f} "
              f"CI {cond['interaction_clinical_minus_radiographic']['ci95']} (n_clin={cond['clinical_test_pos']})")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
