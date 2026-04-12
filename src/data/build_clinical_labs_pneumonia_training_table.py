from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triage-table",
        type=str,
        default="artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore.parquet",
    )
    parser.add_argument(
        "--lab-features",
        type=str,
        default="artifacts/tables/cxr_lab_features.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/logs/cxr_clinical_labs_pneumonia_training_table_u_ignore_report.json",
    )
    args = parser.parse_args()

    triage = pd.read_parquet(args.triage_table).copy()
    labs = pd.read_parquet(args.lab_features).copy()

    key_cols = ["subject_id", "study_id"]

    merged = triage.merge(labs, on=key_cols, how="left")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)

    lab_cols = [
        c for c in labs.columns
        if c not in key_cols
    ]

    report = {
        "rows": int(len(merged)),
        "subjects": int(merged["subject_id"].nunique()),
        "studies": int(merged["study_id"].nunique()),
        "positives": int((merged["target"] == 1).sum()) if "target" in merged.columns else None,
        "negatives": int((merged["target"] == 0).sum()) if "target" in merged.columns else None,
        "num_lab_columns_added": int(len(lab_cols)),
        "lab_coverage_counts": {
            col: int(merged[col].notna().sum())
            for col in lab_cols
            if not col.endswith("_missing")
        },
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()