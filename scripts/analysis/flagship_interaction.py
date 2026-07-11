"""Flagship interaction test: is the clinical fusion gain larger than the radiographic one?

Same patients, same fine-tuned image ensemble and physiology-only triage as
clinical_label_dissociation.py. Resamples patients once per bootstrap draw and computes both
the radiographic and clinical fusion-minus-image deltas on that draw, so their difference (the
interaction) carries a patient-clustered CI.

Output: artifacts/evaluation/clinical_label/flagship_interaction.json
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
SEEDS = [42, 123, 456, 789, 1000]
PHYS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity",
        "temperature_missing", "heartrate_missing", "resprate_missing", "o2sat_missing",
        "sbp_missing", "dbp_missing", "pain_missing", "acuity_missing"]
DIAG = "D:/mimic_iv_ed/diagnosis.csv.gz"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = Path("artifacts/evaluation/clinical_label/flagship_interaction.json")
B = 4000
RNG = np.random.default_rng(20260716)


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


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


def fusion_scores(tr, va, te, Xtr, Xva, Xte, lab):
    tri = LogisticRegression(max_iter=2000).fit(Xtr, tr[lab])
    tv, tt = tri.predict_proba(Xva)[:, 1], tri.predict_proba(Xte)[:, 1]
    meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(va.img), logit(tv)]), va[lab])
    return meta.predict_proba(np.column_stack([logit(te.img), logit(tt)]))[:, 1]


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

    img = te.img.to_numpy()
    yr, yc = te.target.to_numpy(), te.icd.to_numpy()
    fr = fusion_scores(tr, va, te, Xtr, Xva, Xte, "target")
    fc = fusion_scores(tr, va, te, Xtr, Xva, Xte, "icd")

    subj = te.subject_id.to_numpy()
    u = np.unique(subj)
    g = [np.where(subj == x)[0] for x in u]
    dr0 = roc_auc_score(yr, fr) - roc_auc_score(yr, img)
    dc0 = roc_auc_score(yc, fc) - roc_auc_score(yc, img)
    inter, drs, dcs = [], [], []
    for _ in range(B):
        ix = np.concatenate([g[i] for i in RNG.integers(0, len(u), len(u))])
        if len(np.unique(yr[ix])) > 1 and len(np.unique(yc[ix])) > 1:
            dr = roc_auc_score(yr[ix], fr[ix]) - roc_auc_score(yr[ix], img[ix])
            dc = roc_auc_score(yc[ix], fc[ix]) - roc_auc_score(yc[ix], img[ix])
            drs.append(dr); dcs.append(dc); inter.append(dc - dr)
    inter = np.array(inter)
    res = {
        "radiographic_delta": round(float(dr0), 4),
        "clinical_delta": round(float(dc0), 4),
        "interaction_clinical_minus_radiographic": round(float(dc0 - dr0), 4),
        "interaction_ci95": [round(float(np.percentile(inter, 2.5)), 4),
                             round(float(np.percentile(inter, 97.5)), 4)],
        "interaction_p_le0": round(float((inter <= 0).mean()), 4),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
