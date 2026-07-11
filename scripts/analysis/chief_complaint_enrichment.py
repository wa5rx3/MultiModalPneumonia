"""Does enriching the clinical modality grow the clinical gain (and stay neutral radiographically)?

Adds pre-imaging symptom flags parsed from the ED triage chief complaint (cough, dyspnea,
fever, chest pain, sputum, respiratory distress) to the physiological vitals, then compares the
fusion gain over the image for two triage feature sets: vitals only vs vitals + symptoms.

The chief complaint is recorded at triage, before the radiograph, so it is time-safe; we flag
symptoms only and never the suspected diagnosis, so it cannot leak the label the way free-text
"indication" fields can. If enriching the clinical modality grows the clinical fusion gain while
leaving the radiographic label neutral, that is direct evidence the clinical label rewards
clinical context the image does not carry.

Output: artifacts/evaluation/clinical_label/chief_complaint.json
"""
from __future__ import annotations

import json
import re
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
TRIAGE = "D:/mimic_iv_ed/triage.csv.gz"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = Path("artifacts/evaluation/clinical_label/chief_complaint.json")
B = 2000
RNG = np.random.default_rng(20260714)

# symptom lexicon (regex on the lowercased chief complaint); diagnosis terms deliberately excluded
SYMPTOMS = {
    "cough": r"cough",
    "dyspnea": r"\b(sob|dob|dyspnea|dyspnoea|short(ness)? of breath|difficulty breathing|breathing difficulty)\b",
    "fever": r"\b(fever|febrile|chills)\b",
    "chest_pain": r"\b(chest pain|cp|chest discomfort|chest tightness|pleuritic)\b",
    "sputum": r"\b(sputum|productive|phlegm|congestion)\b",
    "resp_distress": r"\b(respiratory distress|resp distress|hypoxia|desat|wheez|tachypnea)\b",
}


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


def symptom_flags(table):
    tri = pd.read_csv(TRIAGE, usecols=["stay_id", "chiefcomplaint"])
    tri["cc"] = tri.chiefcomplaint.fillna("").str.lower()
    for name, pat in SYMPTOMS.items():
        tri[name] = tri.cc.str.contains(pat, regex=True).astype(int)
    flags = tri.set_index("stay_id")[list(SYMPTOMS)]
    merged = table.merge(flags, left_on="stay_id", right_index=True, how="left")
    merged[list(SYMPTOMS)] = merged[list(SYMPTOMS)].fillna(0).astype(int)
    return merged


def boot_delta(y, img, fus, subj):
    u = np.unique(subj)
    g = [np.where(subj == x)[0] for x in u]
    obs = roc_auc_score(y, fus) - roc_auc_score(y, img)
    ds = []
    for _ in range(B):
        ix = np.concatenate([g[i] for i in RNG.integers(0, len(u), len(u))])
        yb = y[ix]
        if len(np.unique(yb)) > 1:
            ds.append(roc_auc_score(yb, fus[ix]) - roc_auc_score(yb, img[ix]))
    ds = np.array(ds)
    return round(float(obs), 4), [round(float(np.percentile(ds, 2.5)), 4),
                                  round(float(np.percentile(ds, 97.5)), 4)], round(float((ds <= 0).mean()), 4)


def fusion_gain(tr, va, te, Xtr, Xva, Xte, lab, subj):
    tri = LogisticRegression(max_iter=3000).fit(Xtr, tr[lab])
    tv, tt = tri.predict_proba(Xva)[:, 1], tri.predict_proba(Xte)[:, 1]
    yv, yt = va[lab].to_numpy(), te[lab].to_numpy()
    meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(va.img), logit(tv)]), yv)
    fus = meta.predict_proba(np.column_stack([logit(te.img), logit(tt)]))[:, 1]
    obs, ci, p = boot_delta(yt, te.img.to_numpy(), fus, subj)
    return {"triage_auroc": round(roc_auc_score(yt, tt), 3), "fusion_auroc": round(roc_auc_score(yt, fus), 3),
            "fusion_minus_image": obs, "ci95": ci, "p_le0": p}


def main():
    d = pd.read_parquet(TABLE)
    d = icd_label(d)
    d = symptom_flags(d)
    syms = list(SYMPTOMS)
    tr = d[d.temporal_split == "train"]
    va = d[d.temporal_split == "validate"].merge(ens("val"), on=KEYS)
    te = d[d.temporal_split == "test"].merge(ens("test"), on=KEYS)
    subj = te.subject_id.to_numpy()

    pre = build_tabular_preprocessor(PHYS, [])
    pre.fit(prepare_tabular_df(tr, PHYS, []))
    Ptr, Pva, Pte = (pre.transform(prepare_tabular_df(x, PHYS, [])) for x in (tr, va, te))
    Str, Sva, Ste = (x[syms].to_numpy() for x in (tr, va, te))
    feat = {"vitals_only": (Ptr, Pva, Pte),
            "vitals_plus_symptoms": (np.hstack([Ptr, Str]), np.hstack([Pva, Sva]), np.hstack([Pte, Ste]))}

    res = {"symptom_prevalence_test": {s: round(float(te[s].mean()), 3) for s in syms}, "results": {}}
    for lab, name in [("target", "radiographic_chexpert"), ("icd", "clinical_icd")]:
        res["results"][name] = {}
        for fname, (Xtr, Xva, Xte) in feat.items():
            res["results"][name][fname] = fusion_gain(tr, va, te, Xtr, Xva, Xte, lab, subj)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    for name, r in res["results"].items():
        v, s = r["vitals_only"], r["vitals_plus_symptoms"]
        print(f"{name}: vitals delta {v['fusion_minus_image']:+.4f} {v['ci95']} -> "
              f"+symptoms delta {s['fusion_minus_image']:+.4f} {s['ci95']}")
    print("symptom prevalence (test):", res["symptom_prevalence_test"])


if __name__ == "__main__":
    main()
