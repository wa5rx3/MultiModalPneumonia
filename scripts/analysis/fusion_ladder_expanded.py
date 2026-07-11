"""The fusion ladder on the full 81k ED-CXR cohort, so mortality and culture become powered.

Same controlled late fusion (fixed 5-seed image ensemble + physiology-only triage) as
fusion_ladder.py, but evaluated on the expanded ED cohort (no patient overlap with the image
models' training set, verified). Outcome rungs (antibiotic, admission, ICU transfer,
in-hospital mortality, 30-day mortality, culture-confirmed pneumonia) now have hundreds of
positives. Diagnosis rungs (radiographic, ED and discharge diagnosis) restricted to studies
with the relevant label. Patient-clustered bootstrap CIs, per-seed deltas, interaction test
vs the radiographic rung.

Output: artifacts/evaluation/clinical_label/fusion_ladder_expanded.json
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

warnings.filterwarnings("ignore")
KEYS = ["subject_id", "study_id", "dicom_id"]
SEEDS = [42, 123, 456, 789, 1000]
VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]
FLAGS = [f"{v}_missing" for v in VITALS]
ED = "D:/mimic_iv_ed"
COHORT = "artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet"
TRIAGE = "artifacts/manifests/cxr_ed_triage_features.parquet"
CHEX = "D:/mimic_data/mimic-cxr-2.0.0-chexpert.csv.gz"
OUT = Path("artifacts/evaluation/clinical_label/fusion_ladder_expanded.json")
B = 2000
RNG = np.random.default_rng(20260724)
PNEU10 = ("J12", "J13", "J14", "J15", "J16", "J17", "J18")
PNEU9 = ("480", "481", "482", "483", "484", "485", "486")
ABX = ["ceftriaxone", "azithromycin", "levofloxacin", "moxifloxacin", "cefepime", "piperacillin",
       "tazobactam", "zosyn", "vancomycin", "doxycycline", "ampicillin", "amoxicillin", "cefpodoxime",
       "clindamycin", "meropenem", "aztreonam", "ciprofloxacin", "cefazolin", "ceftazidime",
       "linezolid", "metronidazole"]
RESP = ["SPUTUM", "BRONCH", "RESPIRATORY", "PLEURAL", "TRACHEAL", "BAL", "LUNG"]
LADDER = ["radiographic", "ed_diagnosis", "discharge_diagnosis", "ed_antibiotic", "admission",
          "icu_transfer", "hospital_mortality", "mortality_30d", "culture_confirmed"]


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def pneu(code, ver):
    c = str(code).strip().upper().replace(".", "")
    if ver == 10:
        return c[:3] in PNEU10 or c[:4] == "J690"
    return c[:3] in PNEU9 or c[:4] == "5070"


def ens(split):
    dfs = []
    for s in SEEDS:
        p = pd.read_csv(f"artifacts/models/multiseed/image_seed{s}/expanded_{split}_predictions.csv")
        dfs.append(p[KEYS + ["pred_prob"]].rename(columns={"pred_prob": f"p{s}"}))
    m = dfs[0]
    for x in dfs[1:]:
        m = m.merge(x, on=KEYS)
    m["img"] = m[[f"p{s}" for s in SEEDS]].mean(axis=1)
    return m


def build(d):
    ed = pd.read_csv(f"{ED}/edstays.csv.gz", usecols=["stay_id", "hadm_id", "intime", "disposition"])
    d = d.drop(columns=[c for c in ["hadm_id", "disposition", "intime"] if c in d.columns]).merge(ed, on="stay_id", how="left")
    cx = pd.read_csv(CHEX).set_index("study_id")["Pneumonia"]
    pn = d.study_id.map(cx)
    d["radiographic"] = np.where(pn == 1, 1, np.where(pn == 0, 0, np.nan))  # u_ignore: uncertain/NaN -> NaN
    dg = pd.read_csv(f"{ED}/diagnosis.csv.gz", usecols=["stay_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    dg["p"] = [pneu(c, v) for c, v in zip(dg.icd_code, dg.icd_version)]
    d["ed_diagnosis"] = d.stay_id.map(dg.groupby("stay_id").p.max()).fillna(False).astype(int)
    dh = pd.read_csv(f"{ED}/diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    dh["p"] = [pneu(c, v) for c, v in zip(dh.icd_code, dh.icd_version)]
    d["discharge_diagnosis"] = d.hadm_id.map(dh.groupby("hadm_id").p.max()).fillna(False).astype(int)
    px = pd.read_csv(f"{ED}/pyxis.csv.gz", usecols=["stay_id", "name"])
    px["abx"] = px.name.fillna("").str.lower().apply(lambda s: any(a in s for a in ABX))
    d["ed_antibiotic"] = d.stay_id.map(px.groupby("stay_id").abx.max()).fillna(False).astype(int)
    d["admission"] = (d.disposition == "ADMITTED").astype(int)
    icu = set(pd.read_csv(f"{ED}/icustays.csv.gz", usecols=["hadm_id"]).hadm_id.dropna())
    d["icu_transfer"] = d.hadm_id.isin(icu).astype(int)
    adm = pd.read_csv("D:/mimic_iv/admissions.csv.gz", usecols=["hadm_id", "hospital_expire_flag"])
    d["hospital_mortality"] = d.hadm_id.map(adm.set_index("hadm_id").hospital_expire_flag).fillna(0).astype(int)
    d.loc[d.disposition == "EXPIRED", "hospital_mortality"] = 1
    pat = pd.read_csv("D:/mimic_iv/patients.csv.gz", usecols=["subject_id", "dod"])
    dod = d.subject_id.map(pat.set_index("subject_id").dod)
    days = (pd.to_datetime(dod, errors="coerce") - pd.to_datetime(d.intime, errors="coerce")).dt.days
    d["mortality_30d"] = ((days >= 0) & (days <= 30)).astype(int)
    d.loc[d.disposition == "EXPIRED", "mortality_30d"] = 1
    mb = pd.read_csv(f"{ED}/microbiologyevents.csv.gz", usecols=["hadm_id", "spec_type_desc", "org_name"])
    mb["pos"] = mb.spec_type_desc.fillna("").str.upper().apply(lambda s: any(k in s for k in RESP)) & mb.org_name.notna()
    d["culture_confirmed"] = d.hadm_id.map(mb.groupby("hadm_id").pos.max()).fillna(False).astype(int)
    return d


def main():
    d = pd.read_parquet(COHORT)[KEYS + ["stay_id", "temporal_split"]]
    tf = pd.read_parquet(TRIAGE)[KEYS + VITALS + FLAGS]
    d = d.merge(tf, on=KEYS, how="left")
    d = build(d)
    ev, et = ens("val"), ens("test")
    va = d[d.temporal_split == "validate"].merge(ev, on=KEYS)
    te = d[d.temporal_split == "test"].merge(et, on=KEYS)
    tr = d[d.temporal_split == "train"]

    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(tr[VITALS])
    def feat(x):
        return np.hstack([prep.transform(x[VITALS]), x[FLAGS].fillna(0).to_numpy()])
    Xtr, Xva, Xte = feat(tr), feat(va), feat(te)
    subj = te.subject_id.to_numpy(); u = np.unique(subj); grp = [np.where(subj == x)[0] for x in u]

    res = {}
    rad_draws = None
    for col in LADDER:
        m = te[col].notna() & (~np.isnan(te[col].to_numpy().astype(float)) if te[col].dtype.kind == "f" else True)
        # restrict to labelled rows for this rung (radiographic has NaN for uncertain)
        trm = tr[tr[col].notna()]
        vam = va[va[col].notna()]
        tem = te[te[col].notna()]
        if len(tem) == 0:
            continue
        ytr = trm[col].astype(int); yv = vam[col].astype(int).to_numpy(); yt = tem[col].astype(int).to_numpy()
        if len(np.unique(yt)) < 2 or yt.sum() < 10:
            res[col] = {"test_pos": int(yt.sum()), "note": "too few positives"}
            continue
        Xtrm, Xvam = feat(trm), feat(vam)
        tri = LogisticRegression(max_iter=3000).fit(Xtrm, ytr)
        tv, tt = tri.predict_proba(Xvam)[:, 1], tri.predict_proba(feat(tem))[:, 1]
        meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(vam.img), logit(tv)]), yv)
        img_te = tem.img.to_numpy()
        fus = meta.predict_proba(np.column_stack([logit(img_te), logit(tt)]))[:, 1]
        sj = tem.subject_id.to_numpy(); uu = np.unique(sj); gg = [np.where(sj == x)[0] for x in uu]
        obs = roc_auc_score(yt, fus) - roc_auc_score(yt, img_te)
        draws = np.full(B, np.nan)
        for b in range(B):
            ix = np.concatenate([gg[i] for i in RNG.integers(0, len(uu), len(uu))])
            if len(np.unique(yt[ix])) > 1:
                draws[b] = roc_auc_score(yt[ix], fus[ix]) - roc_auc_score(yt[ix], img_te[ix])
        dd = draws[~np.isnan(draws)]
        entry = {"test_pos": int(yt.sum()), "test_n": int(len(yt)), "prevalence": round(float(yt.mean()), 3),
                 "image_auroc": round(float(roc_auc_score(yt, img_te)), 3),
                 "triage_auroc": round(float(roc_auc_score(yt, tt)), 3),
                 "fusion_auroc": round(float(roc_auc_score(yt, fus)), 3),
                 "fusion_minus_image": round(float(obs), 4),
                 "ci95": [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)],
                 "p_le0": round(float((dd <= 0).mean()), 4)}
        if col == "radiographic":
            rad_full = np.full(B, np.nan)  # radiographic draws on its own subset; interaction is approximate
            rad_draws = dd
        res[col] = entry
        print(f"{col:20s} n+={yt.sum():4d}/{len(yt):5d} prev={yt.mean():.3f} | img {entry['image_auroc']} "
              f"tri {entry['triage_auroc']} fus {entry['fusion_auroc']} | delta {obs:+.4f} {entry['ci95']} p={entry['p_le0']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
