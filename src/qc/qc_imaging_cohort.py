from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def value_counts_dict(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-manifest",
        type=str,
        default="artifacts/manifests/mimic_cxr_manifest.parquet",
    )
    parser.add_argument(
        "--primary-cohort",
        type=str,
        default="artifacts/manifests/mimic_cxr_primary_frontal_cohort.parquet",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="artifacts/manifests/mimic_cxr_qc_report.json",
    )
    parser.add_argument(
        "--missing-paths-csv",
        type=str,
        default="artifacts/manifests/mimic_cxr_missing_paths.csv",
    )
    args = parser.parse_args()

    raw = pd.read_parquet(args.raw_manifest)
    cohort = pd.read_parquet(args.primary_cohort)

    report = {
        "raw": {
            "rows": int(len(raw)),
            "subjects": int(raw["subject_id"].nunique()),
            "studies": int(raw["study_id"].nunique()),
            "split_counts": value_counts_dict(raw["split"]) if "split" in raw.columns else {},
            "view_counts": value_counts_dict(raw["ViewPosition"]) if "ViewPosition" in raw.columns else {},
            "frontal_rows": int(raw["is_frontal"].sum()) if "is_frontal" in raw.columns else None,
            "pa_rows": int(raw["is_pa"].sum()) if "is_pa" in raw.columns else None,
            "ap_rows": int(raw["is_ap"].sum()) if "is_ap" in raw.columns else None,
            "lateral_rows": int(raw["is_lateral"].sum()) if "is_lateral" in raw.columns else None,
            "single_image_rows": int(raw["has_single_image"].sum()) if "has_single_image" in raw.columns else None,
            "multi_image_rows": int(raw["has_multiple_images"].sum()) if "has_multiple_images" in raw.columns else None,
            "missing_t0": int(raw["t0"].isna().sum()) if "t0" in raw.columns else None,
            "existing_paths": int(raw["exists"].fillna(False).sum()) if "exists" in raw.columns else None,
            "missing_paths": int((~raw["exists"].fillna(False)).sum()) if "exists" in raw.columns else None,
        },
        "primary_frontal": {
            "rows": int(len(cohort)),
            "subjects": int(cohort["subject_id"].nunique()),
            "studies": int(cohort["study_id"].nunique()),
            "split_counts": value_counts_dict(cohort["split"]) if "split" in cohort.columns else {},
            "pa_rows": int(cohort["is_pa"].sum()) if "is_pa" in cohort.columns else None,
            "ap_rows": int(cohort["is_ap"].sum()) if "is_ap" in cohort.columns else None,
            "missing_t0": int(cohort["t0"].isna().sum()) if "t0" in cohort.columns else None,
        },
    }

    if "exists" in raw.columns:
        missing = raw[raw["exists"] == False].copy()
        cols = [
            "subject_id", "study_id", "dicom_id", "split",
            "ViewPosition", "StudyDate", "StudyTime", "image_path"
        ]
        cols = [c for c in cols if c in missing.columns]
        missing = missing[cols]
        Path(args.missing_paths_csv).parent.mkdir(parents=True, exist_ok=True)
        missing.to_csv(args.missing_paths_csv, index=False)

    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()