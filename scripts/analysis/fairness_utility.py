"""Subgroup fairness and prevalence-corrected clinical utility for the clinical-label model.

Clinical (ICD) label, flagship setup (fine-tuned image ensemble + physiology-only triage,
late fusion). Both the image-alone and fusion scores are calibrated to the clinical label on
validation (Platt / meta logistic) so probabilities are comparable.

Fairness: per-subgroup AUROC, and TPR / FPR / ECE at one operating threshold (fixed on
validation for 0.80 sensitivity), by gender and collapsed race; reports the max-min gap.

Utility: decision-curve net benefit for image vs fusion across threshold probabilities, and
positive predictive value at a fixed 0.80-sensitivity operating point evaluated at the observed
and two lower ED base rates (prevalence correction).

Output: artifacts/evaluation/clinical_label/fairness_utility.json
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
OUT = Path("artifacts/evaluation/clinical_label/fairness_utility.json")
TARGET_SENS = 0.80


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


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() > 0:
            e += abs(p[m].mean() - y[m].mean()) * m.mean()
    return float(e)


def thresh_at_sens(y, p, sens):
    pos = np.sort(p[y == 1])
    return float(pos[int((1 - sens) * len(pos))]) if len(pos) else 0.5


def subgroup_metrics(y, p, thr, groups):
    out = {}
    for g in sorted(set(groups)):
        m = groups == g
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        pred = (p[m] >= thr).astype(int)
        tp = ((pred == 1) & (y[m] == 1)).sum(); fn = ((pred == 0) & (y[m] == 1)).sum()
        fp = ((pred == 1) & (y[m] == 0)).sum(); tn = ((pred == 0) & (y[m] == 0)).sum()
        out[g] = {"n": int(m.sum()), "prevalence": round(float(y[m].mean()), 3),
                  "auroc": round(float(roc_auc_score(y[m], p[m])), 3),
                  "tpr": round(float(tp / (tp + fn)) if tp + fn else np.nan, 3),
                  "fpr": round(float(fp / (fp + tn)) if fp + tn else np.nan, 3),
                  "ece": round(ece(y[m], p[m]), 3)}
    return out


def gaps(sub):
    out = {}
    for metric in ["auroc", "tpr", "fpr", "ece"]:
        vals = [v[metric] for v in sub.values() if not np.isnan(v[metric])]
        out[metric + "_gap"] = round(float(max(vals) - min(vals)), 3) if vals else None
    return out


def net_benefit(y, p, pt):
    n = len(y)
    pred = p >= pt
    tp = ((pred) & (y == 1)).sum(); fp = ((pred) & (y == 0)).sum()
    return float(tp / n - (fp / n) * (pt / (1 - pt)))


def ppv_at_prevalence(sens, spec, prev):
    return round(float(sens * prev / (sens * prev + (1 - spec) * (1 - prev))), 3)


def main():
    d = pd.read_parquet(TABLE)
    d = icd_label(d)
    d["race_grp"] = d.race.map(collapse_race)
    pre = build_tabular_preprocessor(PHYS, [])
    tr = d[d.temporal_split == "train"]
    pre.fit(prepare_tabular_df(tr, PHYS, []))
    va = d[d.temporal_split == "validate"].merge(ens("val"), on=KEYS)
    te = d[d.temporal_split == "test"].merge(ens("test"), on=KEYS)
    Xtr = pre.transform(prepare_tabular_df(tr, PHYS, []))
    Xva = pre.transform(prepare_tabular_df(va, PHYS, []))
    Xte = pre.transform(prepare_tabular_df(te, PHYS, []))
    yv, yt = va.icd.to_numpy(), te.icd.to_numpy()

    tri = LogisticRegression(max_iter=2000).fit(Xtr, tr.icd)
    tv, tt = tri.predict_proba(Xva)[:, 1], tri.predict_proba(Xte)[:, 1]
    # calibrated image-alone and fusion probabilities for the clinical label (fit on val)
    imgcal = LogisticRegression(max_iter=1000).fit(logit(va.img).to_numpy().reshape(-1, 1), yv)
    img_p = imgcal.predict_proba(logit(te.img).to_numpy().reshape(-1, 1))[:, 1]
    meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(va.img), logit(tv)]), yv)
    fus_p = meta.predict_proba(np.column_stack([logit(te.img), logit(tt)]))[:, 1]
    img_pv = imgcal.predict_proba(logit(va.img).to_numpy().reshape(-1, 1))[:, 1]
    fus_pv = meta.predict_proba(np.column_stack([logit(va.img), logit(tv)]))[:, 1]

    # fairness at a fixed 0.80-sensitivity threshold set on validation
    res = {"fairness": {}, "utility": {}}
    for mdl, pv, ptest in [("image", img_pv, img_p), ("fusion", fus_pv, fus_p)]:
        thr = thresh_at_sens(yv, pv, TARGET_SENS)
        res["fairness"][mdl] = {}
        for dim, col in [("gender", te.gender.to_numpy()), ("race", te.race_grp.to_numpy())]:
            sub = subgroup_metrics(yt, ptest, thr, col)
            res["fairness"][mdl][dim] = {"subgroups": sub, **gaps(sub)}

    # utility: decision-curve net benefit + prevalence-corrected PPV
    pts = [round(x, 2) for x in np.arange(0.05, 0.51, 0.05)]
    dca = {"threshold_prob": pts, "treat_all": [], "image": [], "fusion": []}
    prev = float(yt.mean())
    for pt in pts:
        dca["treat_all"].append(round(prev - (1 - prev) * (pt / (1 - pt)), 4))
        dca["image"].append(round(net_benefit(yt, img_p, pt), 4))
        dca["fusion"].append(round(net_benefit(yt, fus_p, pt), 4))
    # PPV at 0.80-sensitivity operating point, image vs fusion, across base rates
    ppv = {}
    for mdl, pv, ptest in [("image", img_pv, img_p), ("fusion", fus_pv, fus_p)]:
        thr = thresh_at_sens(yv, pv, TARGET_SENS)
        pred = ptest >= thr
        sens = float(((pred) & (yt == 1)).sum() / (yt == 1).sum())
        spec = float(((~pred) & (yt == 0)).sum() / (yt == 0).sum())
        ppv[mdl] = {"sensitivity": round(sens, 3), "specificity": round(spec, 3),
                    "ppv_at_prevalence": {f"{pr}": ppv_at_prevalence(sens, spec, pr)
                                          for pr in [round(prev, 2), 0.10, 0.05]}}
    res["utility"] = {"observed_clinical_prevalence": round(prev, 3), "decision_curve": dca, "ppv": ppv}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)

    for mdl in ["image", "fusion"]:
        g = res["fairness"][mdl]
        print(f"{mdl}: gender AUROC gap {g['gender']['auroc_gap']}, TPR gap {g['gender']['tpr_gap']}; "
              f"race AUROC gap {g['race']['auroc_gap']}, TPR gap {g['race']['tpr_gap']}")
    print("PPV @0.80 sens:", {m: res["utility"]["ppv"][m]["ppv_at_prevalence"] for m in ["image", "fusion"]})
    print("net benefit @0.10:", "image", dca["image"][pts.index(0.10)], "fusion", dca["fusion"][pts.index(0.10)])


if __name__ == "__main__":
    main()
