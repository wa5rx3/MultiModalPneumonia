from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


NUMERIC_COLS = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "acuity",
]

CATEGORICAL_COLS = [
    "gender",
    "race",
    "arrival_transport",
    # "disposition",  # removed for now: likely post-t0 / leakage risk
]

MISSING_FLAG_COLS = [
    "temperature_missing",
    "heartrate_missing",
    "resprate_missing",
    "o2sat_missing",
    "sbp_missing",
    "dbp_missing",
    "pain_missing",
    "acuity_missing",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/manifests/cxr_ed_triage_features.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/cxr_ed_triage_model_table.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/cxr_ed_triage_model_table_report.json",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input).copy()

    id_cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "hadm_id",
        "stay_id",
        "split",
        "t0",
        "image_path",
        "ViewPosition",
        "is_pa",
        "is_ap",
    ]

    keep_cols = [c for c in id_cols if c in df.columns]
    keep_cols += [c for c in NUMERIC_COLS if c in df.columns]
    keep_cols += [c for c in CATEGORICAL_COLS if c in df.columns]
    keep_cols += [c for c in MISSING_FLAG_COLS if c in df.columns]

    out = df[keep_cols].copy()

    # Force numeric conversion only; do NOT impute here.
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Clean categorical strings, but do NOT use future information
    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
            out[col] = out[col].replace({"": pd.NA})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    report = {
        "rows": int(len(out)),
        "columns": list(out.columns),
        "numeric_missing_counts": {
            col: int(out[col].isna().sum())
            for col in NUMERIC_COLS
            if col in out.columns
        },
        "categorical_missing_counts": {
            col: int(out[col].isna().sum())
            for col in CATEGORICAL_COLS
            if col in out.columns
        },
        "notes": [
            "Numeric values are not imputed in this table.",
            "Imputation must be fit on train only inside model pipelines.",
            "Disposition removed pending time-safety verification.",
        ],
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()