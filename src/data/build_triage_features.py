from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TRIAGE_COLUMNS = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "acuity",
    "chiefcomplaint",
    "gender",
    "race",
    "arrival_transport",
]


def clip_triage_vitals(feat: pd.DataFrame) -> pd.DataFrame:
    # Clip physiological variables to clinically plausible ranges
    # to reduce the effect of measurement artifacts and obvious outliers.
    if "temperature" in feat.columns:
        feat["temperature"] = feat["temperature"].clip(95.0, 105.8)  # Fahrenheit
    if "heartrate" in feat.columns:
        feat["heartrate"] = feat["heartrate"].clip(30, 220)
    if "resprate" in feat.columns:
        feat["resprate"] = feat["resprate"].clip(5, 60)
    if "o2sat" in feat.columns:
        feat["o2sat"] = feat["o2sat"].clip(50, 100)
    if "sbp" in feat.columns:
        feat["sbp"] = feat["sbp"].clip(60, 250)
    if "dbp" in feat.columns:
        feat["dbp"] = feat["dbp"].clip(30, 150)
    return feat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/manifests/cxr_ed_triage_linked.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/cxr_ed_triage_features.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/cxr_ed_triage_features_report.json",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input).copy()

    # Clean duplicate subject_id columns from merge
    if "subject_id_x" in df.columns:
        df = df.rename(columns={"subject_id_x": "subject_id"})
    if "subject_id_y" in df.columns:
        df = df.drop(columns=["subject_id_y"])

    base_cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "hadm_id",
        "stay_id",
        "split",
        "t0",
        "intime",
        "outtime",
        "image_path",
        "ViewPosition",
        "is_pa",
        "is_ap",
    ]

    keep_cols = [c for c in base_cols if c in df.columns]
    keep_cols += [c for c in TRIAGE_COLUMNS if c in df.columns]

    feat = df[keep_cols].copy()

    numeric_cols = [
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "pain",
        "acuity",
    ]

    # Ensure numeric triage fields are numeric before clipping / missing flags
    for col in numeric_cols:
        if col in feat.columns:
            feat[col] = pd.to_numeric(feat[col], errors="coerce")

    feat = clip_triage_vitals(feat)

    for col in numeric_cols:
        if col in feat.columns:
            feat[f"{col}_missing"] = feat[col].isna()

    for col in ["chiefcomplaint", "race", "arrival_transport", "gender"]:
        if col in feat.columns:
            feat[col] = feat[col].astype("string").str.strip()
            feat[col] = feat[col].replace({"": pd.NA})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(args.output, index=False)

    triage_present = pd.Series(True, index=feat.index)
    for col in [
        "temperature",
        "heartrate",
        "resprate",
        "o2sat",
        "sbp",
        "dbp",
        "acuity",
        "chiefcomplaint",
    ]:
        if col in feat.columns:
            triage_present &= feat[col].notna()

    report = {
        "rows": int(len(feat)),
        "subjects": int(feat["subject_id"].nunique()) if "subject_id" in feat.columns else None,
        "studies": int(feat["study_id"].nunique()) if "study_id" in feat.columns else None,
        "stays": int(feat["stay_id"].nunique()) if "stay_id" in feat.columns else None,
        "columns": list(feat.columns),
        "rows_with_all_core_triage_present": int(triage_present.sum()),
        "missing_counts": {
            col: int(feat[col].isna().sum())
            for col in TRIAGE_COLUMNS
            if col in feat.columns
        },
        "notes": [
            "Disposition removed pending time-safety verification.",
            "Physiological variables clipped to clinically plausible ranges.",
        ],
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()