"""Two supporting numbers cited in the text: per-seed deltas and race-only AUROC.

Per-seed: instead of the seed ensemble, use each individual image seed as the image signal
and refit the triage + meta fusion, giving the fusion-minus-image delta per seed for both
labels (mean +/- SD backs the "holds across all five seeds" claim).

Race-only: fit a logistic model on race alone and score each label, to back the statement
that demographics carry no signal here.

Output: artifacts/evaluation/clinical_label/supplements.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
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
OUT = Path("artifacts/evaluation/clinical_label/supplements.json")


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


def preds(seed, split):
    return pd.read_csv(f"artifacts/models/multiseed/image_seed{seed}/{split}_predictions.csv")[
        KEYS + ["pred_prob"]].rename(columns={"pred_prob": "img"})


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


def main():
    d = pd.read_parquet(TABLE)
    d = icd_label(d)
    pre = build_tabular_preprocessor(PHYS, [])
    tr = d[d.temporal_split == "train"]
    pre.fit(prepare_tabular_df(tr, PHYS, []))
    Xtr = pre.transform(prepare_tabular_df(tr, PHYS, []))
    tri_models = {lab: LogisticRegression(max_iter=2000).fit(Xtr, tr[lab]) for lab in ("target", "icd")}

    per_seed = {"radiographic": [], "clinical": []}
    for s in SEEDS:
        va = d[d.temporal_split == "validate"].merge(preds(s, "val"), on=KEYS)
        te = d[d.temporal_split == "test"].merge(preds(s, "test"), on=KEYS)
        Xva = pre.transform(prepare_tabular_df(va, PHYS, []))
        Xte = pre.transform(prepare_tabular_df(te, PHYS, []))
        for lab, name in [("target", "radiographic"), ("icd", "clinical")]:
            tri = tri_models[lab]
            tv = tri.predict_proba(Xva)[:, 1]
            tt = tri.predict_proba(Xte)[:, 1]
            meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(va.img), logit(tv)]), va[lab])
            fus = meta.predict_proba(np.column_stack([logit(te.img), logit(tt)]))[:, 1]
            y = te[lab].to_numpy()
            per_seed[name].append(roc_auc_score(y, fus) - roc_auc_score(y, te.img.to_numpy()))

    # race-only AUROC
    enc = OneHotEncoder(handle_unknown="ignore")
    Rtr = enc.fit_transform(tr[["race"]].astype(str))
    Rte = enc.transform(d[d.temporal_split == "test"][["race"]].astype(str))
    te0 = d[d.temporal_split == "test"]
    race = {}
    for lab, name in [("target", "radiographic"), ("icd", "clinical")]:
        m = LogisticRegression(max_iter=1000).fit(Rtr, tr[lab])
        race[name] = round(float(roc_auc_score(te0[lab], m.predict_proba(Rte)[:, 1])), 3)

    res = {
        "per_seed_delta": {k: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4),
                               "values": [round(float(x), 4) for x in v]} for k, v in per_seed.items()},
        "race_only_auroc": race,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
