"""Build a training table whose `target` is the clinical (ICD) pneumonia label.

Copies the u_ignore temporal training table and replaces `target` with the ED-diagnosis ICD
pneumonia label (same definition as clinical_label_dissociation.py), preserving every other
column and the patient-level temporal split. This lets the existing image-only and concat
trainers retrain end-to-end on the clinical target, so the dissociation can be confirmed with
early fusion rather than only the fixed-image late fusion.

Output: artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal_ICD.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SRC = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
DIAG = "D:/mimic_iv_ed/diagnosis.csv.gz"
OUT = Path("artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal_ICD.parquet")


def is_pneu(code, ver):
    c = str(code).strip().upper().replace(".", "")
    if ver == 10:
        return c[:3] in {"J12", "J13", "J14", "J15", "J16", "J17", "J18"} or c[:4] == "J690"
    return c[:3] in {"480", "481", "482", "483", "484", "485", "486"} or c[:4] == "5070"


def main():
    d = pd.read_parquet(SRC).copy()
    diag = pd.read_csv(DIAG, usecols=["stay_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    diag["pneu"] = [is_pneu(c, v) for c, v in zip(diag.icd_code, diag.icd_version)]
    icd = diag.groupby("stay_id").pneu.max()
    d["target_radiographic"] = d["target"]
    d["target"] = d.stay_id.map(icd).fillna(False).astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT)
    for sp in ["train", "validate", "test"]:
        s = d[d.temporal_split == sp]
        print(f"{sp}: n={len(s)} clinical_prev={s.target.mean():.3f} "
              f"(radiographic_prev={s.target_radiographic.mean():.3f})")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
