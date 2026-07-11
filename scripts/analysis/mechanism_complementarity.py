"""Why the dissociation happens: mechanism (which vitals) + complementarity (which cases).

Uses the flagship pneumonia setup (fine-tuned 5-seed image ensemble + physiology-only
triage), matching clinical_label_dissociation.json.

Mechanism: standardized triage-logistic coefficients for the clinical (ICD) vs radiographic
(CheXpert) label, showing which physiological signals carry clinical information that is
absent for the image-derived label.

Complementarity: a pairwise-concordance decomposition of the AUROC change. Over all
positive/negative test pairs, count the image-incorrect pairs that fusion fixes (vitals add
ranking information the image lacks) versus the image-correct pairs that fusion breaks. A
complementary modality fixes more than it breaks; a redundant one roughly ties.

Output: artifacts/evaluation/clinical_label/mechanism_complementarity.json
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
NICE = {"temperature": "temperature", "heartrate": "heart rate", "resprate": "respiratory rate",
        "o2sat": "oxygen saturation", "sbp": "systolic BP", "dbp": "diastolic BP",
        "pain": "pain score", "acuity": "triage acuity"}
DIAG = "D:/mimic_iv_ed/diagnosis.csv.gz"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
OUT = Path("artifacts/evaluation/clinical_label/mechanism_complementarity.json")


def logit(p, e=1e-6):
    p = np.clip(p, e, 1 - e)
    return np.log(p / (1 - p))


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


def ens(split):
    dfs = [pd.read_csv(f"artifacts/models/multiseed/image_seed{s}/{split}_predictions.csv")
           [KEYS + ["pred_prob"]].rename(columns={"pred_prob": f"p{s}"}) for s in SEEDS]
    m = dfs[0]
    for x in dfs[1:]:
        m = m.merge(x, on=KEYS)
    m["img"] = m[[f"p{s}" for s in SEEDS]].mean(axis=1)
    return m[KEYS + ["img"]]


def concordance_decomposition(y, img, fus):
    """Pairwise ranking decomposition of the fusion-minus-image AUROC change.

    Over every positive/negative pair, a pair is image-correct if the image ranks the
    positive above the negative. Report how many image-incorrect pairs fusion fixes and how
    many image-correct pairs it breaks (ties count as half). fixes - breaks over the pair
    count equals the AUROC change, so the split shows whether the gain is complementary.
    """
    pos, neg = img[y == 1], img[y == 0]
    fpos, fneg = fus[y == 1], fus[y == 0]
    di = pos[:, None] - neg[None, :]          # image margin per pair
    df = fpos[:, None] - fneg[None, :]         # fusion margin per pair
    score_i = np.where(di > 0, 1.0, np.where(di < 0, 0.0, 0.5))
    score_f = np.where(df > 0, 1.0, np.where(df < 0, 0.0, 0.5))
    npairs = score_i.size
    img_wrong = score_i < 1.0                  # image missed or tied
    img_right = score_i == 1.0
    fixed = (score_f[img_wrong] - score_i[img_wrong]).sum()      # gain on image-wrong pairs
    broke = (score_i[img_right] - score_f[img_right]).sum()      # loss on image-right pairs
    return {
        "n_pairs": int(npairs),
        "auroc_image": round(float(score_i.mean()), 4),
        "auroc_fusion": round(float(score_f.mean()), 4),
        "frac_pairs_image_wrong": round(float(img_wrong.mean()), 4),
        "fixes_per_pair": round(float(fixed / npairs), 4),
        "breaks_per_pair": round(float(broke / npairs), 4),
        "net_gain": round(float((fixed - broke) / npairs), 4),
        "fix_to_break_ratio": round(float(fixed / broke), 2) if broke > 0 else None,
    }


def main():
    d = pd.read_parquet(TABLE)
    d = icd_label(d)
    pre = build_tabular_preprocessor(PHYS, [])
    tr = d[d.temporal_split == "train"]
    pre.fit(prepare_tabular_df(tr, PHYS, []))
    imv, imt = ens("val"), ens("test")
    va = d[d.temporal_split == "validate"].merge(imv, on=KEYS)
    te = d[d.temporal_split == "test"].merge(imt, on=KEYS)
    Xtr = pre.transform(prepare_tabular_df(tr, PHYS, []))
    Xva = pre.transform(prepare_tabular_df(va, PHYS, []))
    Xte = pre.transform(prepare_tabular_df(te, PHYS, []))

    res = {"labels": {}}
    for lab, name in [("target", "radiographic_chexpert"), ("icd", "clinical_icd")]:
        tri = LogisticRegression(max_iter=2000).fit(Xtr, tr[lab])
        tv = tri.predict_proba(Xva)[:, 1]
        tt = tri.predict_proba(Xte)[:, 1]
        yt = te[lab].to_numpy()
        meta = LogisticRegression(max_iter=1000).fit(np.column_stack([logit(va.img), logit(tv)]), va[lab])
        fus = meta.predict_proba(np.column_stack([logit(te.img), logit(tt)]))[:, 1]
        # mechanism: standardized coefficients (preprocessor already scales numerics)
        coefs = {NICE.get(f, f): round(float(c), 3)
                 for f, c in zip(PHYS, tri.coef_[0]) if not f.endswith("_missing")}
        top = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))
        res["labels"][name] = {
            "triage_auroc": round(roc_auc_score(yt, tt), 3),
            "image_auroc": round(roc_auc_score(yt, te.img), 3),
            "fusion_auroc": round(roc_auc_score(yt, fus), 3),
            "triage_coefficients": dict(top),
            "complementarity": concordance_decomposition(yt, te.img.to_numpy(), fus),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    for name, r in res["labels"].items():
        c = r["complementarity"]
        print(f"{name}: fusion {r['fusion_auroc']} vs image {r['image_auroc']}; "
              f"fixes/breaks per pair {c['fixes_per_pair']}/{c['breaks_per_pair']} "
              f"(ratio {c['fix_to_break_ratio']}); top vitals {list(r['triage_coefficients'])[:3]}")


if __name__ == "__main__":
    main()
