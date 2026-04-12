from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-table",
        type=str,
        default="artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only.parquet",
    )
    parser.add_argument(
        "--lab-features",
        type=str,
        default="artifacts/tables/cxr_lab_features_hadm_only.parquet",
    )
    parser.add_argument(
        "--output-table",
        type=str,
        default="artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_overlap.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/logs/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_overlap_report.json",
    )
    args = parser.parse_args()

    full_df = pd.read_parquet(args.input_table).copy()
    labs_df = pd.read_parquet(args.lab_features).copy()

    key_cols = ["subject_id", "study_id"]
    overlap_keys = labs_df[key_cols].drop_duplicates()

    out = full_df.merge(overlap_keys, on=key_cols, how="inner")

    Path(args.output_table).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_table, index=False)

    report = {
        "input_rows": int(len(full_df)),
        "overlap_rows": int(len(out)),
        "subjects": int(out["subject_id"].nunique()),
        "studies": int(out["study_id"].nunique()),
        "positives": int((out["target"] == 1).sum()) if "target" in out.columns else None,
        "negatives": int((out["target"] == 0).sum()) if "target" in out.columns else None,
        "positive_rate": float(out["target"].mean()) if "target" in out.columns and len(out) > 0 else None,
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()