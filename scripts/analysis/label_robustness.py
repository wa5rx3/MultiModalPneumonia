"""Is the dissociation robust to how the clinical (ICD) pneumonia label is defined?

Recomputes the clinical fusion gain (flagship setup) under several ICD code-set definitions of
pneumonia, from strict (bacterial/viral pneumonia only) to broad (adding influenza-with-pneumonia
and aspiration). If the clinical gain and its significance hold across definitions, the finding is
not an artifact of one code list.

Output: artifacts/evaluation/clinical_label/label_robustness.json
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
OUT = Path("artifacts/evaluation/clinical_label/label_robustness.json")
B = 2000
RNG = np.random.default_rng(20260715)

# each definition: (icd10 3-char prefixes, icd10 4-char extras, icd9 3-char prefixes, icd9 4-char extras)
DEFS = {
    "strict_bacterial_viral": ({"J12", "J13", "J14", "J15", "J16", "J17", "J18"}, set(),
                               {"480", "481", "482", "483", "484", "485", "486"}, set()),
    "primary": ({"J12", "J13", "J14", "J15", "J16", "J17", "J18"}, {"J690"},
                {"480", "481", "482", "483", "484", "485", "486"}, {"5070"}),
    "broad_with_influenza": ({"J09", "J10", "J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18"}, {"J690"},
                             {"480", "481", "482", "483", "484", "485", "486", "487", "488"}, {"5070"}),
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


def label_col(diag, p10, e10, p9, e9):
    def hit(c, v):
        c = str(c).strip().upper().replace(".", "")
        return (c[:3] in p10 or c[:4] in e10) if v == 10 else (c[:3] in p9 or c[:4] in e9)
    diag["h"] = [hit(c, v) for c, v in zip(diag.icd_code, diag.icd_version)]
    return diag.groupby("stay_id").h.max()


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


def main():
    d = pd.read_parquet(TABLE)
    diag = pd.read_csv(DIAG, usecols=["stay_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    pre = build_tabular_preprocessor(PHYS, [])
    tr0 = d[d.temporal_split == "train"]
    pre.fit(prepare_tabular_df(tr0, PHYS, []))
    va = d[d.temporal_split == "validate"].merge(ens("val"), on=KEYS)
    te = d[d.temporal_split == "test"].merge(ens("test"), on=KEYS)
    Xtr = pre.transform(prepare_tabular_df(tr0, PHYS, []))
    Xva = pre.transform(prepare_tabular_df(va, PHYS, []))
    Xte = pre.transform(prepare_tabular_df(te, PHYS, []))
    subj = te.subject_id.to_numpy()

    res = {}
    for defname, (p10, e10, p9, e9) in DEFS.items():
        lc = label_col(diag.copy(), p10, e10, p9, e9)
        ytr = tr0.stay_id.map(lc).fillna(False).astype(int)
        yv = va.stay_id.map(lc).fillna(False).astype(int).to_numpy()
        yt = te.stay_id.map(lc).fillna(False).astype(int).to_numpy()
        tri = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        tv, tt = tri.predict_proba(Xva)[:, 1], tri.predict_proba(Xte)[:, 1]
        meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(va.img), logit(tv)]), yv)
        fus = meta.predict_proba(np.column_stack([logit(te.img), logit(tt)]))[:, 1]
        obs, ci, p = boot_delta(yt, te.img.to_numpy(), fus, subj)
        res[defname] = {"test_positives": int(yt.sum()), "test_prevalence": round(float(yt.mean()), 3),
                        "image_auroc": round(roc_auc_score(yt, te.img), 3),
                        "fusion_auroc": round(roc_auc_score(yt, fus), 3),
                        "fusion_minus_image": obs, "ci95": ci, "p_le0": p}
        print(f"{defname:24s} n+={yt.sum():4d} prev={yt.mean():.3f} delta {obs:+.4f} {ci} p={p}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
