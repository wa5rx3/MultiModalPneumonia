"""Does the value of triage fusion grow as the target moves from radiograph to outcome?

Same controlled late fusion (fixed image pneumonia ensemble + physiology-only triage) scored
against a ladder of targets on the same ED cohort:
  radiographic sign (CheXpert)  ->  clinical diagnosis (ED ICD)  ->  hospital admission
  ->  short-term mortality.
Outcome labels come from the previously unused MIMIC-IV hospital module: edstays.disposition
(admitted), admissions.hospital_expire_flag (in-hospital death), patients.dod (30-day death).
Reports the fusion-minus-image AUROC change per target with patient-clustered bootstrap CIs.

Output: artifacts/evaluation/clinical_label/outcome_gradient.json
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
EDSTAYS = "D:/mimic_iv_ed/edstays.csv.gz"
ADMIT = "D:/mimic_iv/admissions.csv.gz"
PATId = "D:/mimic_iv/patients.csv.gz"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = Path("artifacts/evaluation/clinical_label/outcome_gradient.json")
B = 2000
RNG = np.random.default_rng(20260722)


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


def add_labels(d):
    # clinical ICD pneumonia
    dg = pd.read_csv(DIAG, usecols=["stay_id", "icd_code", "icd_version"], dtype={"icd_code": str})

    def isp(c, v):
        c = str(c).strip().upper().replace(".", "")
        if v == 10:
            return c[:3] in {"J12", "J13", "J14", "J15", "J16", "J17", "J18"} or c[:4] == "J690"
        return c[:3] in {"480", "481", "482", "483", "484", "485", "486"} or c[:4] == "5070"

    dg["pneu"] = [isp(c, v) for c, v in zip(dg.icd_code, dg.icd_version)]
    d["icd"] = d.stay_id.map(dg.groupby("stay_id").pneu.max()).fillna(False).astype(int)

    ed = pd.read_csv(EDSTAYS, usecols=["stay_id", "hadm_id", "subject_id", "intime", "disposition"])
    ed["admitted"] = (ed.disposition == "ADMITTED").astype(int)
    ed["ed_expired"] = (ed.disposition == "EXPIRED").astype(int)
    d = d.drop(columns=[c for c in ["hadm_id", "intime", "disposition"] if c in d.columns])
    d = d.merge(ed[["stay_id", "hadm_id", "intime", "admitted", "ed_expired"]], on="stay_id", how="left")

    adm = pd.read_csv(ADMIT, usecols=["hadm_id", "hospital_expire_flag"])
    d = d.merge(adm, on="hadm_id", how="left")
    d["hospital_mortality"] = d.hospital_expire_flag.fillna(0).astype(int)
    # counts ED death as a death too
    d.loc[d.ed_expired == 1, "hospital_mortality"] = 1

    pat = pd.read_csv(PATId, usecols=["subject_id", "dod"])
    d = d.merge(pat, on="subject_id", how="left")
    it = pd.to_datetime(d.intime, errors="coerce")
    dod = pd.to_datetime(d.dod, errors="coerce")
    days = (dod - it).dt.days
    d["died_30d"] = ((days >= 0) & (days <= 30)).astype(int)
    d.loc[d.ed_expired == 1, "died_30d"] = 1
    return d


def boot(y, img, fus, subj):
    u = np.unique(subj); g = [np.where(subj == x)[0] for x in u]
    obs = roc_auc_score(y, fus) - roc_auc_score(y, img)
    ds = []
    for _ in range(B):
        ix = np.concatenate([g[i] for i in RNG.integers(0, len(u), len(u))])
        yb = y[ix]
        if len(np.unique(yb)) > 1:
            ds.append(roc_auc_score(yb, fus[ix]) - roc_auc_score(yb, img[ix]))
    ds = np.array(ds)
    return round(float(obs), 4), [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)], round(float((ds <= 0).mean()), 4)


def main():
    d = pd.read_parquet(TABLE)
    d = add_labels(d)
    pre = build_tabular_preprocessor(PHYS, [])
    tr = d[d.temporal_split == "train"]
    pre.fit(prepare_tabular_df(tr, PHYS, []))
    va = d[d.temporal_split == "validate"].merge(ens("val"), on=KEYS)
    te = d[d.temporal_split == "test"].merge(ens("test"), on=KEYS)
    Xtr = pre.transform(prepare_tabular_df(tr, PHYS, []))
    Xva = pre.transform(prepare_tabular_df(va, PHYS, []))
    Xte = pre.transform(prepare_tabular_df(te, PHYS, []))
    subj = te.subject_id.to_numpy()

    ladder = [("target", "radiographic"), ("icd", "clinical"),
              ("admitted", "admission"), ("hospital_mortality", "hospital_mortality"),
              ("died_30d", "mortality_30d")]
    res = {}
    for col, name in ladder:
        ytr, yv, yt = tr[col], va[col].to_numpy(), te[col].to_numpy()
        if yt.sum() < 10 or len(np.unique(yt)) < 2:
            res[name] = {"skipped": f"only {int(yt.sum())} positives"}
            continue
        tri = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        tv, tt = tri.predict_proba(Xva)[:, 1], tri.predict_proba(Xte)[:, 1]
        meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(va.img), logit(tv)]), yv)
        fus = meta.predict_proba(np.column_stack([logit(te.img), logit(tt)]))[:, 1]
        obs, ci, p = boot(yt, te.img.to_numpy(), fus, subj)
        res[name] = {"test_pos": int(yt.sum()), "prevalence": round(float(yt.mean()), 3),
                     "image_auroc": round(float(roc_auc_score(yt, te.img)), 3),
                     "triage_auroc": round(float(roc_auc_score(yt, tt)), 3),
                     "fusion_auroc": round(float(roc_auc_score(yt, fus)), 3),
                     "fusion_minus_image": obs, "ci95": ci, "p_le0": p}
        print(f"{name:18s} n+={yt.sum():4d} prev={yt.mean():.3f} | image {res[name]['image_auroc']} "
              f"triage {res[name]['triage_auroc']} fusion {res[name]['fusion_auroc']} | "
              f"delta {obs:+.4f} {ci} p={p}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
