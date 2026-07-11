"""Bootstrap CIs for the label-provenance fairness finding.

For the fixed image ensemble, resample patients (clustered) and recompute, for each label,
the subgroup AUROCs and the sex AUROC gap, plus the cross-label difference of the sex gap.
Also report per-race-group AUROCs with CIs to judge whether the worst-group flip is real or
small-sample noise.

Output: artifacts/evaluation/clinical_label/fairness_by_label_ci.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
KEYS = ["subject_id", "study_id", "dicom_id"]
SEEDS = [42, 123, 456, 789, 1000]
DIAG = "D:/mimic_iv_ed/diagnosis.csv.gz"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = Path("artifacts/evaluation/clinical_label/fairness_by_label_ci.json")
B = 2000
RNG = np.random.default_rng(20260720)


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


def collapse_race(r):
    r = str(r).upper()
    if r.startswith("WHITE"):
        return "White"
    if r.startswith("BLACK"):
        return "Black"
    if "HISPANIC" in r or "LATINO" in r:
        return "Hispanic"
    if r.startswith("ASIAN"):
        return "Asian"
    return "Other"


def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def ci(a):
    a = a[~np.isnan(a)]
    return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]


def main():
    d = pd.read_parquet(TABLE)
    d = icd_label(d)
    d["race_grp"] = d.race.map(collapse_race)
    te = d[d.temporal_split == "test"].merge(ens("test"), on=KEYS)
    img = te.img.to_numpy()
    sex = te.gender.to_numpy()
    race = te.race_grp.to_numpy()
    subj = te.subject_id.to_numpy()
    u = np.unique(subj)
    groups = [np.where(subj == x)[0] for x in u]
    yr, yc = te.target.to_numpy(), te.icd.to_numpy()
    races = ["White", "Black", "Hispanic", "Asian", "Other"]

    # bootstrap
    sex_gap = {"radiographic": [], "clinical": []}   # AUROC(F) - AUROC(M)
    gap_diff = []                                     # radiographic gap - clinical gap
    race_auc = {lab: {r: [] for r in races} for lab in ("radiographic", "clinical")}
    for _ in range(B):
        ix = np.concatenate([groups[i] for i in RNG.integers(0, len(u), len(u))])
        for lab, y in [("radiographic", yr), ("clinical", yc)]:
            f = (sex[ix] == "F"); m = (sex[ix] == "M")
            gf = safe_auc(y[ix][f], img[ix][f]); gm = safe_auc(y[ix][m], img[ix][m])
            sex_gap[lab].append(gf - gm)
            for r in races:
                rm = race[ix] == r
                race_auc[lab][r].append(safe_auc(y[ix][rm], img[ix][rm]))
        gap_diff.append((sex_gap["radiographic"][-1]) - (sex_gap["clinical"][-1]))

    def obs_gap(y):
        return safe_auc(y[sex == "F"], img[sex == "F"]) - safe_auc(y[sex == "M"], img[sex == "M"])

    res = {
        "sex_auroc_gap_F_minus_M": {
            "radiographic": {"estimate": round(float(obs_gap(yr)), 3), "ci95": ci(np.array(sex_gap["radiographic"]))},
            "clinical": {"estimate": round(float(obs_gap(yc)), 3), "ci95": ci(np.array(sex_gap["clinical"]))},
            "cross_label_difference": {"estimate": round(float(obs_gap(yr) - obs_gap(yc)), 3),
                                       "ci95": ci(np.array(gap_diff)),
                                       "p_le0": round(float((np.array(gap_diff) <= 0).mean()), 4)},
        },
        "race_auroc_by_label": {lab: {r: {"n": int((race == r).sum()),
                                          "estimate": round(float(safe_auc(y[race == r], img[race == r])), 3)
                                          if (race == r).sum() and len(np.unique(y[race == r])) > 1 else None,
                                          "ci95": ci(np.array(race_auc[lab][r]))}
                                      for r in races}
                                for lab, y in [("radiographic", yr), ("clinical", yc)]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    s = res["sex_auroc_gap_F_minus_M"]
    print(f"SEX AUROC gap (F-M): radiographic {s['radiographic']['estimate']} {s['radiographic']['ci95']}; "
          f"clinical {s['clinical']['estimate']} {s['clinical']['ci95']}")
    print(f"  cross-label difference {s['cross_label_difference']['estimate']} "
          f"{s['cross_label_difference']['ci95']} p={s['cross_label_difference']['p_le0']}")
    for lab in ("radiographic", "clinical"):
        print(f"RACE AUROC {lab}: " + ", ".join(
            f"{r} {res['race_auroc_by_label'][lab][r]['estimate']}{res['race_auroc_by_label'][lab][r]['ci95']}(n{res['race_auroc_by_label'][lab][r]['n']})"
            for r in races))


if __name__ == "__main__":
    main()
