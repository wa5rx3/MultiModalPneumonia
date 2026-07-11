"""Confirm the dissociation is not specific to late fusion.

The primary dissociation uses a fixed image model and a validation-fit late-fusion
meta-learner. Here we retrain end to end: image-only and concat (early feature fusion) models
are trained separately on each label, and we compare concat with image-only against the same
label. If early fusion also improves the clinical label but not the radiographic one, the
dissociation is a property of the target, not of the fusion mechanism.

Image-only and concat are seed-42 runs trained directly on each label. Patient-clustered
bootstrap CI on the concat-minus-image AUROC change.

Output: artifacts/evaluation/clinical_label/early_fusion.json
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
OUT = Path("artifacts/evaluation/clinical_label/early_fusion.json")
B = 2000
RNG = np.random.default_rng(20260718)

RUNS = {
    "clinical": {
        "image": "artifacts/models/clinical_label/image_icd_seed42/test_predictions.csv",
        "concat": "artifacts/models/clinical_label/concat_icd_seed42/test_predictions.csv",
    },
    "radiographic": {
        "image": "artifacts/models/multiseed/image_seed42/test_predictions.csv",
        "concat": "artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv",
    },
}


def boot_delta(y, img, con, subj):
    u = np.unique(subj)
    g = [np.where(subj == x)[0] for x in u]
    obs = roc_auc_score(y, con) - roc_auc_score(y, img)
    ds = []
    for _ in range(B):
        ix = np.concatenate([g[i] for i in RNG.integers(0, len(u), len(u))])
        yb = y[ix]
        if len(np.unique(yb)) > 1:
            ds.append(roc_auc_score(yb, con[ix]) - roc_auc_score(yb, img[ix]))
    ds = np.array(ds)
    return obs, ds


def main():
    res = {}
    deltas = {}
    for label, paths in RUNS.items():
        im = pd.read_csv(paths["image"])[KEYS + ["target", "pred_prob"]].rename(columns={"pred_prob": "img"})
        co = pd.read_csv(paths["concat"])[KEYS + ["pred_prob"]].rename(columns={"pred_prob": "con"})
        m = im.merge(co, on=KEYS)
        y = m.target.to_numpy()
        obs, ds = boot_delta(y, m.img.to_numpy(), m.con.to_numpy(), m.subject_id.to_numpy())
        deltas[label] = ds
        res[label] = {
            "n": int(len(m)), "prevalence": round(float(y.mean()), 3),
            "image_auroc": round(float(roc_auc_score(y, m.img)), 4),
            "concat_auroc": round(float(roc_auc_score(y, m.con)), 4),
            "concat_minus_image": round(float(obs), 4),
            "ci95": [round(float(np.percentile(ds, 2.5)), 4), round(float(np.percentile(ds, 97.5)), 4)],
            "p_le0": round(float((ds <= 0).mean()), 4),
        }
        print(f"{label}: image {res[label]['image_auroc']} concat {res[label]['concat_auroc']} "
              f"delta {res[label]['concat_minus_image']:+.4f} {res[label]['ci95']} p={res[label]['p_le0']}")

    n = min(len(deltas["clinical"]), len(deltas["radiographic"]))
    inter = deltas["clinical"][:n] - deltas["radiographic"][:n]
    res["interaction_clinical_minus_radiographic"] = {
        "estimate": round(float(res["clinical"]["concat_minus_image"] - res["radiographic"]["concat_minus_image"]), 4),
        "ci95": [round(float(np.percentile(inter, 2.5)), 4), round(float(np.percentile(inter, 97.5)), 4)],
        "note": "radiographic concat is a single existing run, not seed-matched to the clinical concat",
    }
    print("interaction:", res["interaction_clinical_minus_radiographic"]["estimate"],
          res["interaction_clinical_minus_radiographic"]["ci95"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
