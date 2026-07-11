"""Count positives for every candidate ladder rung in the cohort, to see what is powered.

Rungs (image -> outcome): radiographic sign, ED presenting diagnosis, hospital discharge
diagnosis, culture-confirmed pneumonia, ED antibiotic, admission, ICU transfer, mortality.
Reports test-set positive counts and prevalence so underpowered rungs are excluded up front.
No modelling here, only label construction and counts.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ED = "D:/mimic_iv_ed"
TABLE = "artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet"
PNEU10 = ("J12", "J13", "J14", "J15", "J16", "J17", "J18")
PNEU9 = ("480", "481", "482", "483", "484", "485", "486")
# CAP / sepsis antibiotics commonly given in the ED
ABX = ["ceftriaxone", "azithromycin", "levofloxacin", "moxifloxacin", "cefepime",
       "piperacillin", "tazobactam", "zosyn", "vancomycin", "doxycycline", "ampicillin",
       "amoxicillin", "cefpodoxime", "clindamycin", "meropenem", "aztreonam", "ciprofloxacin",
       "cefazolin", "ceftazidime", "linezolid", "metronidazole"]
RESP_SPEC = ["SPUTUM", "BRONCH", "RESPIRATORY", "PLEURAL", "TRACHEAL", "BAL", "LUNG"]


def pneu_icd(code, ver):
    c = str(code).strip().upper().replace(".", "")
    if ver == 10:
        return c[:3] in PNEU10 or c[:4] == "J690"
    return c[:3] in PNEU9 or c[:4] == "5070"


def main():
    d = pd.read_parquet(TABLE)[["subject_id", "stay_id", "study_id", "dicom_id", "temporal_split", "target"]]
    ed = pd.read_csv(f"{ED}/edstays.csv.gz", usecols=["stay_id", "hadm_id", "intime", "disposition"])
    d = d.merge(ed, on="stay_id", how="left")

    lab = {"radiographic": d.target.astype(int)}

    # ED presenting diagnosis pneumonia
    dg = pd.read_csv(f"{ED}/diagnosis.csv.gz", usecols=["stay_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    dg["p"] = [pneu_icd(c, v) for c, v in zip(dg.icd_code, dg.icd_version)]
    lab["ed_diagnosis"] = d.stay_id.map(dg.groupby("stay_id").p.max()).fillna(False).astype(int)

    # hospital discharge diagnosis pneumonia (by hadm_id)
    dh = pd.read_csv(f"{ED}/diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"], dtype={"icd_code": str})
    dh["p"] = [pneu_icd(c, v) for c, v in zip(dh.icd_code, dh.icd_version)]
    lab["discharge_diagnosis"] = d.hadm_id.map(dh.groupby("hadm_id").p.max()).fillna(False).astype(int)

    # culture-confirmed pneumonia: respiratory specimen with an organism, by hadm_id
    mb = pd.read_csv(f"{ED}/microbiologyevents.csv.gz",
                     usecols=["hadm_id", "spec_type_desc", "org_name"])
    mb["resp"] = mb.spec_type_desc.fillna("").str.upper().apply(lambda s: any(k in s for k in RESP_SPEC))
    mb["pos"] = mb.resp & mb.org_name.notna()
    lab["culture_confirmed"] = d.hadm_id.map(mb.groupby("hadm_id").pos.max()).fillna(False).astype(int)

    # ED antibiotic given (by ED stay_id)
    px = pd.read_csv(f"{ED}/pyxis.csv.gz", usecols=["stay_id", "name"])
    px["abx"] = px.name.fillna("").str.lower().apply(lambda s: any(a in s for a in ABX))
    lab["ed_antibiotic"] = d.stay_id.map(px.groupby("stay_id").abx.max()).fillna(False).astype(int)

    # admission and ICU transfer
    lab["admission"] = (d.disposition == "ADMITTED").astype(int)
    icu = pd.read_csv(f"{ED}/icustays.csv.gz", usecols=["hadm_id"])
    icu_set = set(icu.hadm_id.dropna())
    lab["icu_transfer"] = d.hadm_id.isin(icu_set).astype(int)

    df = pd.DataFrame(lab)
    df["temporal_split"] = d.temporal_split.values
    te = df[df.temporal_split == "test"]
    print(f"cohort: total {len(df)}, test {len(te)}\n")
    print(f"{'rung':22s} {'test_pos':>8s} {'test_prev':>9s} {'train_pos':>9s}")
    for c in ["radiographic", "ed_diagnosis", "discharge_diagnosis", "culture_confirmed",
              "ed_antibiotic", "admission", "icu_transfer"]:
        tp = int(te[c].sum()); pr = te[c].mean()
        trp = int(df[df.temporal_split == "train"][c].sum())
        print(f"{c:22s} {tp:8d} {pr:9.3f} {trp:9d}")


if __name__ == "__main__":
    main()
