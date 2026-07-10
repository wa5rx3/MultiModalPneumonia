"""Radiographic vs clinical pneumonia label: does triage fusion help?

Builds a non-radiographic pneumonia label from the MIMIC-IV-ED discharge diagnosis
(ICD-9 480-486 / ICD-10 J12-J18), and tests whether adding triage vitals to the image
model helps predict it, versus the CheXpert (radiographic) label. Late fusion: the
seed-ensembled image model's score is combined with a triage logistic model by a
meta-learner fit on validation. Patient-level bootstrap CIs on the fusion-image delta.

Output: artifacts/evaluation/clinical_label/dissociation.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.training.train_multimodal_pneumonia import (
    TRIAGE_NUMERIC_COLS, TRIAGE_CATEGORICAL_COLS, build_tabular_preprocessor, prepare_tabular_df)

warnings.filterwarnings("ignore")
KEYS = ["subject_id", "study_id", "dicom_id"]
SEEDS = [42, 123, 456, 789, 1000]
# physiology-only triage features (race/gender excluded: demographics carry no signal here,
# AUROC 0.52, and using race to predict a diagnosis raises a fairness confound)
PHYS_NUM = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity",
            "temperature_missing", "heartrate_missing", "resprate_missing", "o2sat_missing",
            "sbp_missing", "dbp_missing", "pain_missing", "acuity_missing"]
PHYS_CAT = []
DIAG = "D:/mimic_iv_ed/diagnosis.csv.gz"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = Path("artifacts/evaluation/clinical_label/dissociation.json")
B = 2000
RNG = np.random.default_rng(20260712)


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def icd_pneumonia_label(table):
    d = pd.read_csv(DIAG, usecols=["stay_id", "icd_code", "icd_version"], dtype={"icd_code": str})

    def is_pneu(code, ver):
        c = str(code).strip().upper().replace(".", "")
        if ver == 10:
            return c[:3] in {"J12", "J13", "J14", "J15", "J16", "J17", "J18"} or c[:4] == "J690"
        return c[:3] in {"480", "481", "482", "483", "484", "485", "486"} or c[:4] == "5070"

    d["pneu"] = [is_pneu(c, v) for c, v in zip(d.icd_code, d.icd_version)]
    sp = d.groupby("stay_id").pneu.max()
    table["icd"] = table.stay_id.map(sp).fillna(False).astype(int)
    return table, set(d.stay_id.unique())


def ens(split):
    dfs = [pd.read_csv(f"artifacts/models/multiseed/image_seed{s}/{split}_predictions.csv")
           [KEYS + ["pred_prob"]].rename(columns={"pred_prob": f"p{s}"}) for s in SEEDS]
    m = dfs[0]
    for x in dfs[1:]:
        m = m.merge(x, on=KEYS)
    m["img"] = m[[f"p{s}" for s in SEEDS]].mean(axis=1)
    return m[KEYS + ["img"]]


def boot_delta(y, img, fus, subj):
    uniq = np.unique(subj)
    groups = [np.where(subj == u)[0] for u in uniq]
    obs = roc_auc_score(y, fus) - roc_auc_score(y, img)
    ds = np.empty(B)
    for b in range(B):
        ix = np.concatenate([groups[i] for i in RNG.integers(0, len(uniq), len(uniq))])
        yb = y[ix]
        ds[b] = np.nan if yb.min() == yb.max() else roc_auc_score(yb, fus[ix]) - roc_auc_score(yb, img[ix])
    ds = ds[~np.isnan(ds)]
    return float(obs), [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))], float((ds <= 0).mean())


def main():
    d = pd.read_parquet(TABLE)
    d, stays = icd_pneumonia_label(d)
    pre = build_tabular_preprocessor(PHYS_NUM, PHYS_CAT)
    tr = d[d.temporal_split == "train"]
    pre.fit(prepare_tabular_df(tr, PHYS_NUM, PHYS_CAT))
    imv, imt = ens("val"), ens("test")
    vak = d[d.temporal_split == "validate"].merge(imv, on=KEYS)
    tek = d[d.temporal_split == "test"].merge(imt, on=KEYS)
    Xtr = pre.transform(prepare_tabular_df(tr, PHYS_NUM, PHYS_CAT))
    Xva = pre.transform(prepare_tabular_df(vak, PHYS_NUM, PHYS_CAT))
    Xte = pre.transform(prepare_tabular_df(tek, PHYS_NUM, PHYS_CAT))

    res = {"icd_coverage": f"{100*d.stay_id.isin(stays).mean():.1f}%",
           "icd_test_prevalence": round(float(tek.icd.mean()), 3),
           "chexpert_test_prevalence": round(float(tek.target.mean()), 3),
           "labels": {}}
    for lab, name in [("target", "radiographic_chexpert"), ("icd", "clinical_icd")]:
        tri = LogisticRegression(max_iter=2000).fit(Xtr, tr[lab])
        tv, tt = tri.predict_proba(Xva)[:, 1], tri.predict_proba(Xte)[:, 1]
        yv, yt = vak[lab].to_numpy(), tek[lab].to_numpy()
        meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(vak.img), logit(tv)]), yv)
        fus = meta.predict_proba(np.column_stack([logit(tek.img), logit(tt)]))[:, 1]
        obs, ci, p_le0 = boot_delta(yt, tek.img.to_numpy(), fus, tek.subject_id.to_numpy())
        res["labels"][name] = {
            "image_auroc": round(roc_auc_score(yt, tek.img), 3),
            "triage_auroc": round(roc_auc_score(yt, tt), 3),
            "fusion_auroc": round(roc_auc_score(yt, fus), 3),
            "fusion_minus_image": round(obs, 4),
            "delta_95ci": [round(x, 4) for x in ci],
            "p_delta_le_0": round(p_le0, 4),
        }
    json.dump(res, open(OUT.parent.mkdir(parents=True, exist_ok=True) or OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
