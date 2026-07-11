"""The fusion ladder: does the value of triage fusion climb from radiograph to outcome?

Same controlled late fusion (fixed five-seed image pneumonia ensemble + physiology-only triage,
meta-learner fit on validation) scored against a ladder of targets on the same ED cohort,
ordered by increasing distance from the radiograph:
  radiographic sign -> ED diagnosis -> discharge diagnosis -> ED antibiotic -> admission
  -> ICU transfer.
Reports image/triage/fusion AUROC and the fusion-minus-image change with patient-clustered
bootstrap CIs, per-seed deltas for robustness, and an interaction test of each rung's delta
against the radiographic delta. Culture-confirmed pneumonia and mortality are reported
separately as underpowered.

Output: artifacts/evaluation/clinical_label/fusion_ladder.json
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
ED = "D:/mimic_iv_ed"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = Path("artifacts/evaluation/clinical_label/fusion_ladder.json")
B = 2000
RNG = np.random.default_rng(20260723)
PNEU10 = ("J12", "J13", "J14", "J15", "J16", "J17", "J18")
PNEU9 = ("480", "481", "482", "483", "484", "485", "486")
ABX = ["ceftriaxone", "azithromycin", "levofloxacin", "moxifloxacin", "cefepime", "piperacillin",
       "tazobactam", "zosyn", "vancomycin", "doxycycline", "ampicillin", "amoxicillin", "cefpodoxime",
       "clindamycin", "meropenem", "aztreonam", "ciprofloxacin", "cefazolin", "ceftazidime",
       "linezolid", "metronidazole"]
RESP = ["SPUTUM", "BRONCH", "RESPIRATORY", "PLEURAL", "TRACHEAL", "BAL", "LUNG"]
LADDER = ["radiographic", "ed_diagnosis", "discharge_diagnosis", "ed_antibiotic", "admission", "icu_transfer"]
UNDERPOWERED = ["culture_confirmed", "hospital_mortality"]


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def pneu(code, ver):
    c = str(code).strip().upper().replace(".", "")
    if ver == 10:
        return c[:3] in PNEU10 or c[:4] == "J690"
    return c[:3] in PNEU9 or c[:4] == "5070"


def ens(split):
    dfs = [pd.read_csv(f"artifacts/models/multiseed/image_seed{s}/{split}_predictions.csv")
           [KEYS + ["pred_prob"]].rename(columns={"pred_prob": f"p{s}"}) for s in SEEDS]
    m = dfs[0]
    for x in dfs[1:]:
        m = m.merge(x, on=KEYS)
    m["img"] = m[[f"p{s}" for s in SEEDS]].mean(axis=1)
    return m[KEYS + ["img"] + [f"p{s}" for s in SEEDS]]


def build_labels(d):
    ed = pd.read_csv(f"{ED}/edstays.csv.gz", usecols=["stay_id", "hadm_id", "disposition"])
    d = d.drop(columns=[c for c in ["hadm_id", "disposition"] if c in d.columns]).merge(ed, on="stay_id", how="left")
    d["radiographic"] = d.target.astype(int)
    dg = pd.read_csv(f"{ED}/diagnosis.csv.gz", usecols=["stay_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    dg["p"] = [pneu(c, v) for c, v in zip(dg.icd_code, dg.icd_version)]
    d["ed_diagnosis"] = d.stay_id.map(dg.groupby("stay_id").p.max()).fillna(False).astype(int)
    dh = pd.read_csv(f"{ED}/diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    dh["p"] = [pneu(c, v) for c, v in zip(dh.icd_code, dh.icd_version)]
    d["discharge_diagnosis"] = d.hadm_id.map(dh.groupby("hadm_id").p.max()).fillna(False).astype(int)
    mb = pd.read_csv(f"{ED}/microbiologyevents.csv.gz", usecols=["hadm_id", "spec_type_desc", "org_name"])
    mb["pos"] = mb.spec_type_desc.fillna("").str.upper().apply(lambda s: any(k in s for k in RESP)) & mb.org_name.notna()
    d["culture_confirmed"] = d.hadm_id.map(mb.groupby("hadm_id").pos.max()).fillna(False).astype(int)
    px = pd.read_csv(f"{ED}/pyxis.csv.gz", usecols=["stay_id", "name"])
    px["abx"] = px.name.fillna("").str.lower().apply(lambda s: any(a in s for a in ABX))
    d["ed_antibiotic"] = d.stay_id.map(px.groupby("stay_id").abx.max()).fillna(False).astype(int)
    d["admission"] = (d.disposition == "ADMITTED").astype(int)
    icu = set(pd.read_csv(f"{ED}/icustays.csv.gz", usecols=["hadm_id"]).hadm_id.dropna())
    d["icu_transfer"] = d.hadm_id.isin(icu).astype(int)
    adm = pd.read_csv("D:/mimic_iv/admissions.csv.gz", usecols=["hadm_id", "hospital_expire_flag"])
    d["hospital_mortality"] = d.hadm_id.map(adm.set_index("hadm_id").hospital_expire_flag).fillna(0).astype(int)
    return d


def fit_fusion(tr, va, te, Xtr, Xva, Xte, col, imcol="img"):
    tri = LogisticRegression(max_iter=2000).fit(Xtr, tr[col])
    tv, tt = tri.predict_proba(Xva)[:, 1], tri.predict_proba(Xte)[:, 1]
    meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(va[imcol]), logit(tv)]), va[col])
    fus = meta.predict_proba(np.column_stack([logit(te[imcol]), logit(tt)]))[:, 1]
    return tt, fus


def main():
    d = pd.read_parquet(TABLE)
    d = build_labels(d)
    pre = build_tabular_preprocessor(PHYS, [])
    tr = d[d.temporal_split == "train"]
    pre.fit(prepare_tabular_df(tr, PHYS, []))
    va = d[d.temporal_split == "validate"].merge(ens("val"), on=KEYS)
    te = d[d.temporal_split == "test"].merge(ens("test"), on=KEYS)
    Xtr = pre.transform(prepare_tabular_df(tr, PHYS, []))
    Xva = pre.transform(prepare_tabular_df(va, PHYS, []))
    Xte = pre.transform(prepare_tabular_df(te, PHYS, []))
    subj = te.subject_id.to_numpy()
    u = np.unique(subj); groups = [np.where(subj == x)[0] for x in u]
    img = te.img.to_numpy()

    # radiographic delta bootstrap draws kept for the interaction test
    rad_draws = None
    res = {}
    for col in LADDER + UNDERPOWERED:
        yt = te[col].to_numpy()
        if len(np.unique(yt)) < 2:
            continue
        tt, fus = fit_fusion(tr, va, te, Xtr, Xva, Xte, col)
        obs = roc_auc_score(yt, fus) - roc_auc_score(yt, img)
        draws = np.full(B, np.nan)
        for b in range(B):
            ix = np.concatenate([groups[i] for i in RNG.integers(0, len(u), len(u))])
            yb = yt[ix]
            if len(np.unique(yb)) > 1:
                draws[b] = roc_auc_score(yb, fus[ix]) - roc_auc_score(yb, img[ix])
        dd = draws[~np.isnan(draws)]
        entry = {"test_pos": int(yt.sum()), "prevalence": round(float(yt.mean()), 3),
                 "image_auroc": round(float(roc_auc_score(yt, img)), 3),
                 "triage_auroc": round(float(roc_auc_score(yt, tt)), 3),
                 "fusion_auroc": round(float(roc_auc_score(yt, fus)), 3),
                 "fusion_minus_image": round(float(obs), 4),
                 "ci95": [round(float(np.percentile(dd, 2.5)), 4), round(float(np.percentile(dd, 97.5)), 4)],
                 "p_le0": round(float((dd <= 0).mean()), 4)}
        if col == "radiographic":
            rad_draws = draws
        elif rad_draws is not None:
            inter = (draws - rad_draws)
            inter = inter[~np.isnan(inter)]
            entry["interaction_vs_radiographic"] = {
                "estimate": round(float(obs - res["radiographic"]["fusion_minus_image"]), 4),
                "ci95": [round(float(np.percentile(inter, 2.5)), 4), round(float(np.percentile(inter, 97.5)), 4)],
                "p_le0": round(float((inter <= 0).mean()), 4)}
        # per-seed for well-powered rungs
        if col in LADDER:
            per = []
            for s in SEEDS:
                tts, fs = fit_fusion(tr, va, te, Xtr, Xva, Xte, col, imcol=f"p{s}")
                per.append(roc_auc_score(yt, fs) - roc_auc_score(yt, te[f"p{s}"].to_numpy()))
            entry["per_seed_delta"] = {"mean": round(float(np.mean(per)), 4), "sd": round(float(np.std(per)), 4)}
        res[col] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"{'rung':20s} {'n+':>4s} {'img':>5s} {'tri':>5s} {'fus':>5s} {'delta':>8s} {'ci':>20s} {'inter_p':>8s}")
    for col in LADDER + UNDERPOWERED:
        if col not in res:
            continue
        e = res[col]
        ip = e.get("interaction_vs_radiographic", {}).get("p_le0", "-")
        print(f"{col:20s} {e['test_pos']:4d} {e['image_auroc']:.3f} {e['triage_auroc']:.3f} {e['fusion_auroc']:.3f} "
              f"{e['fusion_minus_image']:+.4f} {str(e['ci95']):>20s} {ip}")


if __name__ == "__main__":
    main()
