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
        default="artifacts/manifests/cxr_ed_triage_model_table.parquet",
    )
    parser.add_argument(
        "--label-table",
        type=str,
        default="artifacts/manifests/cxr_pneumonia_training_table_u_ignore.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_report.json",
    )
    args = parser.parse_args()

    triage = pd.read_parquet(args.triage_table).copy()
    labels = pd.read_parquet(args.label_table).copy()

    key_cols = ["subject_id", "study_id"]
    keep_label_cols = [c for c in key_cols + ["target", "split"] if c in labels.columns]
    labels = labels[keep_label_cols].drop_duplicates()

    merged = triage.merge(labels, on=key_cols, how="inner", suffixes=("", "_label"))

    # prefer label split if duplicate appears
    if "split_label" in merged.columns:
        merged["split"] = merged["split_label"]
        merged = merged.drop(columns=["split_label"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)

    report = {
        "rows": int(len(merged)),
        "subjects": int(merged["subject_id"].nunique()),
        "studies": int(merged["study_id"].nunique()),
        "positives": int((merged["target"] == 1).sum()),
        "negatives": int((merged["target"] == 0).sum()),
        "positive_rate": float(merged["target"].mean()) if len(merged) > 0 else None,
        "split_counts": {
            str(k): int(v) for k, v in merged["split"].value_counts(dropna=False).to_dict().items()
        } if "split" in merged.columns else {},
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()