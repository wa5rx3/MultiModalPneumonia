from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ed-temporal-cohort",
        type=str,
        default="artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet",
    )
    parser.add_argument(
        "--label-table",
        type=str,
        default="artifacts/manifests/cxr_pneumonia_training_table_u_ignore.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/cxr_image_pneumonia_finetune_table_u_ignore_temporal_report.json",
    )
    args = parser.parse_args()

    cohort = pd.read_parquet(args.ed_temporal_cohort).copy()
    labels = pd.read_parquet(args.label_table).copy()

    required_cohort = [
        "subject_id",
        "study_id",
        "dicom_id",
        "image_path",
        "t0",
        "temporal_split",
        "is_pa",
        "is_ap",
    ]
    missing_cohort = [c for c in required_cohort if c not in cohort.columns]
    if missing_cohort:
        raise ValueError(f"ED temporal cohort missing columns: {missing_cohort}")

    required_labels = [
        "subject_id",
        "study_id",
        "pneumonia_chexpert_raw",
        "pneumonia_positive",
        "pneumonia_negative",
    ]
    missing_labels = [c for c in required_labels if c not in labels.columns]
    if missing_labels:
        raise ValueError(f"Label table missing columns: {missing_labels}")

    keep_cols = [
        "subject_id",
        "study_id",
        "pneumonia_chexpert_raw",
        "pneumonia_positive",
        "pneumonia_negative",
    ]
    if "pneumonia_uncertain" in labels.columns:
        keep_cols.append("pneumonia_uncertain")
    if "target" in labels.columns:
        keep_cols.append("target")
    labels = labels[keep_cols].copy()




    if "target" in labels.columns:
        labels["target"] = pd.to_numeric(labels["target"], errors="raise")
    else:
        labels["target"] = pd.NA
        labels.loc[labels["pneumonia_positive"] == True, "target"] = 1
        labels.loc[labels["pneumonia_negative"] == True, "target"] = 0

        if "pneumonia_uncertain" in labels.columns:
            labels.loc[labels["pneumonia_uncertain"] == True, "target"] = 0

    merged = cohort.merge(
        labels,
        on=["subject_id", "study_id"],
        how="inner",
        validate="one_to_one",
    )

    missing_target = int(merged["target"].isna().sum())
    if missing_target > 0:
        raise ValueError(
            f"Label table produced {missing_target} missing target values after merge. "
            "Ensure the label-policy table contains a fully-defined binary target."
        )
    merged["target"] = pd.to_numeric(merged["target"], errors="raise").astype(int)
    merged["pneumonia_chexpert_raw"] = pd.to_numeric(
        merged["pneumonia_chexpert_raw"], errors="coerce"
    )

    output_cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "image_path",
        "t0",
        "temporal_split",
        "is_pa",
        "is_ap",
        "pneumonia_chexpert_raw",
        "target",
    ]
    out = merged[output_cols].copy()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    report = {
        "rows": int(len(out)),
        "subjects": int(out["subject_id"].nunique()),
        "studies": int(out["study_id"].nunique()),
        "positives": int((out["target"] == 1).sum()),
        "negatives": int((out["target"] == 0).sum()),
        "positive_rate": float(out["target"].mean()),
        "temporal_split_row_counts": {
            k: int(v)
            for k, v in out["temporal_split"].value_counts().to_dict().items()
        },
        "temporal_split_subject_counts": {
            k: int(v)
            for k, v in out.groupby("temporal_split")["subject_id"].nunique().to_dict().items()
        },
        "notes": [
            "Image fine-tuning table is built from the ED temporal cohort plus current binary pneumonia label-policy table.",
            "One frontal image per study is inherited from the upstream ED cohort.",
            "Temporal split is the sacred evaluation split.",
        ],
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()