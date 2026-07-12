"""Multi-seed Section 4.3: retraining the image model on the clinical (ICD) label.

Addresses the single-seed vulnerability of the early-fusion control. For each of five seeds an
image-only model and a concat (early-fusion) model were retrained end to end on the ICD label
(see artifacts/models/clinical_label/{image,concat}_icd_seed*). This reads their test
predictions (target = ICD label) and reports the multi-seed clinical-label AUROC of each,
against the fixed radiographic-trained references (image 0.784, late fusion 0.808).

Output: artifacts/evaluation/clinical_label/early_fusion_multiseed.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
SEEDS = [42, 123, 456, 789, 1000]
OUT = Path("artifacts/evaluation/clinical_label/early_fusion_multiseed.json")


def auroc(path):
    d = pd.read_csv(path)
    return roc_auc_score(d.target, d.pred_prob)


def main():
    img = [auroc(f"artifacts/models/clinical_label/image_icd_seed{s}/test_predictions.csv") for s in SEEDS]
    con = [auroc(f"artifacts/models/clinical_label/concat_icd_seed{s}/test_predictions.csv") for s in SEEDS]
    img, con = np.array(img), np.array(con)
    res = {
        "seeds": SEEDS,
        "fixed_radiographic_image_on_clinical": 0.784,
        "fixed_image_plus_triage_late_fusion": 0.808,
        "image_retrained_on_clinical": {"mean": round(float(img.mean()), 4), "sd": round(float(img.std()), 4),
                                        "values": [round(float(x), 4) for x in img]},
        "concat_retrained_on_clinical": {"mean": round(float(con.mean()), 4), "sd": round(float(con.std()), 4),
                                         "values": [round(float(x), 4) for x in con]},
        "concat_minus_retrained_image": round(float(con.mean() - img.mean()), 4),
        "triage_marginal_on_fixed_model": round(0.808 - 0.784, 4),
        "triage_marginal_on_retrained_model": round(float(con.mean() - img.mean()), 4),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
