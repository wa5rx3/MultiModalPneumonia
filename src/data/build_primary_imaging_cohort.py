from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        type=str,
        default="artifacts/manifests/mimic_cxr_manifest.parquet",
    )
    parser.add_argument(
        "--output-cohort",
        type=str,
        default="artifacts/manifests/mimic_cxr_primary_frontal_cohort.parquet",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="artifacts/manifests/mimic_cxr_primary_frontal_cohort_report.json",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input_manifest).copy()

    # Keep only rows with valid paths.
    # exists=True  → verified present, keep.
    # exists=False → verified missing, drop.
    # exists=NA   → not checked, assume present.
    if "exists" in df.columns and not df["exists"].isna().all():
        df = df[df["exists"] == True].copy()

    # Keep only frontal rows
    df = df[df["is_frontal"] == True].copy()

    # Selection priority: PA first, then AP
    df["view_priority"] = 1
    df.loc[df["is_ap"] == True, "view_priority"] = 2
    df.loc[df["is_pa"] == True, "view_priority"] = 1

    # Stable sort so first row per study is deterministic
    sort_cols = ["subject_id", "study_id", "view_priority", "dicom_id"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols).copy()

    # Keep one image per study
    cohort = (
        df.groupby(["subject_id", "study_id"], as_index=False)
        .head(1)
        .copy()
    )

    # Clean up
    if "view_priority" in cohort.columns:
        cohort = cohort.drop(columns=["view_priority"])

    output_path = Path(args.output_cohort)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(output_path, index=False)

    report = {
        "input_rows": int(len(pd.read_parquet(args.input_manifest))),
        "rows_after_exists_filter": int(len(pd.read_parquet(args.input_manifest).query("exists == True"))) if "exists" in pd.read_parquet(args.input_manifest).columns else None,
        "final_rows": int(len(cohort)),
        "final_subjects": int(cohort["subject_id"].nunique()),
        "final_studies": int(cohort["study_id"].nunique()),
        "final_pa_rows": int(cohort["is_pa"].sum()) if "is_pa" in cohort.columns else None,
        "final_ap_rows": int(cohort["is_ap"].sum()) if "is_ap" in cohort.columns else None,
        "missing_t0": int(cohort["t0"].isna().sum()) if "t0" in cohort.columns else None,
    }

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()