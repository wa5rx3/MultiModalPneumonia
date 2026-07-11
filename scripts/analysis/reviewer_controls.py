"""Decisive controls demanded by adversarial review of the dissociation.

Same flagship setup as clinical_label_dissociation.py (fixed 5-seed image ensemble +
physiology-only triage, meta-learner on val, patient-clustered bootstrap on test).

1. PREVALENCE MATCH: the radiographic label is 45% prevalent, the clinical label 19%. Is the
   interaction a base-rate artifact? Subsample radiographic positives to 19% prevalence, recompute
   the radiographic fusion delta many times, and compare to the clinical delta.
2. CODING PROPENSITY (acuity): the triage-acuity coefficient jumps for the clinical label; acuity
   is an administrative severity score that predicts workup/admission/coding. Recompute the clinical
   fusion delta with full triage, triage minus acuity, and acuity only. If acuity-only reproduces
   the gain, the effect may be coding propensity rather than physiology.
3. ACUITY STRATA and ADMITTED-ONLY: does the clinical gain survive within acuity strata and among
   admitted patients only (where coding is near-universal)?

Output: artifacts/evaluation/clinical_label/reviewer_controls.json
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
VITALS = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]
FULL = VITALS + [f"{v}_missing" for v in VITALS]
NO_ACUITY = [c for c in FULL if "acuity" not in c]
ACUITY_ONLY = ["acuity", "acuity_missing"]
DIAG = "D:/mimic_iv_ed/diagnosis.csv.gz"
EDSTAYS = "D:/mimic_iv_ed/edstays.csv.gz"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = Path("artifacts/evaluation/clinical_label/reviewer_controls.json")
B = 2000
RNG = np.random.default_rng(20260812)


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


def fusion_scores(tr, va, te, cols, label):
    pre = build_tabular_preprocessor(cols, [])
    Xtr = pre.fit_transform(prepare_tabular_df(tr, cols, []))
    Xva = pre.transform(prepare_tabular_df(va, cols, []))
    Xte = pre.transform(prepare_tabular_df(te, cols, []))
    tri = LogisticRegression(max_iter=2000).fit(Xtr, tr[label])
    tv, tt = tri.predict_proba(Xva)[:, 1], tri.predict_proba(Xte)[:, 1]
    meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(va.img), logit(tv)]), va[label])
    fus = meta.predict_proba(np.column_stack([logit(te.img), logit(tt)]))[:, 1]
    return tt, fus


def clustered_delta(y, img, fus, subj):
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
    d = icd_label(d)
    ed = pd.read_csv(EDSTAYS, usecols=["stay_id", "disposition"])
    d = d.drop(columns=[c for c in ["disposition"] if c in d.columns]).merge(ed, on="stay_id", how="left")
    tr = d[d.temporal_split == "train"]
    va = d[d.temporal_split == "validate"].merge(ens("val"), on=KEYS)
    te = d[d.temporal_split == "test"].merge(ens("test"), on=KEYS)
    subj = te.subject_id.to_numpy()
    img = te.img.to_numpy()
    res = {}

    # baseline fusion scores (full triage) for each label
    ttr, fusr = fusion_scores(tr, va, te, FULL, "target")     # radiographic
    ttc, fusc = fusion_scores(tr, va, te, FULL, "icd")        # clinical
    yr, yc = te.target.to_numpy(), te.icd.to_numpy()
    dr, cir, _ = clustered_delta(yr, img, fusr, subj)
    dc, cic, pc = clustered_delta(yc, img, fusc, subj)
    res["baseline"] = {"radiographic_delta": dr, "radiographic_ci": cir,
                       "clinical_delta": dc, "clinical_ci": cic, "clinical_p": pc,
                       "triage_alone_radiographic": round(float(roc_auc_score(yr, ttr)), 3),
                       "triage_alone_clinical": round(float(roc_auc_score(yc, ttc)), 3)}

    # 1. PREVALENCE MATCH: subsample radiographic positives to 19% prevalence
    pos = np.where(yr == 1)[0]; neg = np.where(yr == 0)[0]
    target_prev = float(yc.mean())
    n_pos_keep = int(round(target_prev * len(neg) / (1 - target_prev)))
    matched = []
    for _ in range(1000):
        keep = np.concatenate([RNG.choice(pos, n_pos_keep, replace=False), neg])
        matched.append(roc_auc_score(yr[keep], fusr[keep]) - roc_auc_score(yr[keep], img[keep]))
    matched = np.array(matched)
    res["prevalence_matched"] = {
        "target_prevalence": round(target_prev, 3), "radiographic_pos_kept": n_pos_keep,
        "radiographic_delta_at_19pct": round(float(matched.mean()), 4),
        "ci95": [round(float(np.percentile(matched, 2.5)), 4), round(float(np.percentile(matched, 97.5)), 4)],
        "interaction_clinical_minus_matched_radiographic": round(float(dc - matched.mean()), 4),
        "note": "if the matched radiographic delta stays near zero, the interaction is not a base-rate artifact"}

    # 2. CODING PROPENSITY: clinical delta with full / no-acuity / acuity-only triage
    res["acuity_ablation"] = {}
    for name, cols in [("full_triage", FULL), ("triage_minus_acuity", NO_ACUITY), ("acuity_only", ACUITY_ONLY)]:
        _, fus = fusion_scores(tr, va, te, cols, "icd")
        dd, ci, p = clustered_delta(yc, img, fus, subj)
        res["acuity_ablation"][name] = {"clinical_delta": dd, "ci95": ci, "p_le0": p}

    # 3. ACUITY STRATA and ADMITTED-ONLY (using the full-triage clinical fusion scores)
    res["subsets"] = {}
    acu = te.acuity.fillna(te.acuity.median()).to_numpy()
    strata = {"high_acuity(<=2)": acu <= 2, "low_acuity(>=3)": acu >= 3,
              "admitted_only": (te.disposition == "ADMITTED").to_numpy()}
    for name, mask in strata.items():
        if mask.sum() < 30 or len(np.unique(yc[mask])) < 2:
            res["subsets"][name] = {"n": int(mask.sum()), "note": "too few"}
            continue
        dd, ci, p = clustered_delta(yc[mask], img[mask], fusc[mask], subj[mask])
        res["subsets"][name] = {"n": int(mask.sum()), "clinical_pos": int(yc[mask].sum()),
                                "clinical_delta": dd, "ci95": ci, "p_le0": p}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
