"""Build an image evaluation table from non-ED MIMIC-CXR frontal studies.

Excludes all subjects present in the ED cohort, merges CheXpert pneumonia
labels, and applies u_ignore policy (pos/neg only). Adds a synthetic
temporal_split="eval" column so CXRBinaryDataset can be used without changes.

Note: The DenseNet backbone was pretrained on non-ED MIMIC-CXR studies, so
this is an internal generalization check, not a truly external validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary-frontal-cohort",
        type=str,
        default="artifacts/manifests/mimic_cxr_primary_frontal_cohort.parquet",
    )
    parser.add_argument(
        "--ed-cohort",
        type=str,
        default="artifacts/manifests/cxr_final_ed_cohort.parquet",
    )
    parser.add_argument(
        "--chexpert-labels",
        type=str,
        default=None,
        help="Path to mimic-cxr-2.0.0-chexpert.csv.gz (required)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/nonED_image_eval_table.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/nonED_image_eval_table_report.json",
    )
    args = parser.parse_args()

    frontal = pd.read_parquet(args.primary_frontal_cohort)
    ed_cohort = pd.read_parquet(args.ed_cohort)

    ed_subject_ids = set(ed_cohort["subject_id"].unique())
    non_ed = frontal[~frontal["subject_id"].isin(ed_subject_ids)].copy()
    print(f"Primary frontal cohort: {len(frontal):,} rows")
    print(f"ED subjects excluded: {len(ed_subject_ids):,}")
    print(f"Non-ED rows remaining: {len(non_ed):,}")

    chexpert = pd.read_csv(args.chexpert_labels, usecols=["subject_id", "study_id", "Pneumonia"])
    chexpert = chexpert.rename(columns={"Pneumonia": "pneumonia_chexpert"})

    merged = non_ed.merge(chexpert, on=["subject_id", "study_id"], how="left")

    # u_ignore policy: keep only definite pos (1.0) or neg (0.0)
    labeled = merged[merged["pneumonia_chexpert"].isin([1.0, 0.0])].copy()
    labeled["target"] = (labeled["pneumonia_chexpert"] == 1.0).astype(int)

    # Synthetic split label so CXRBinaryDataset filters correctly
    labeled["temporal_split"] = "eval"

    keep_cols = ["subject_id", "study_id", "dicom_id", "image_path", "target", "temporal_split"]
    available = [c for c in keep_cols if c in labeled.columns]
    out = labeled[available].reset_index(drop=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    report = {
        "rows": int(len(out)),
        "subjects": int(out["subject_id"].nunique()),
        "positives": int((out["target"] == 1).sum()),
        "negatives": int((out["target"] == 0).sum()),
        "positive_rate": float(out["target"].mean()),
        "note": (
            "Non-ED MIMIC-CXR internal generalization. "
            "DenseNet backbone was pretrained on this population — "
            "label this accordingly in the thesis."
        ),
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
