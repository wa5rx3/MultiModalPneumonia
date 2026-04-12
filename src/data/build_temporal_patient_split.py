from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/manifests/cxr_final_ed_cohort.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/cxr_final_ed_cohort_with_temporal_split_report.json",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input).copy()
    df["t0"] = pd.to_datetime(df["t0"], errors="coerce")

    patient_times = (
        df.groupby("subject_id", as_index=False)["t0"]
        .min()
        .rename(columns={"t0": "patient_first_t0"})
        .sort_values("patient_first_t0")
        .reset_index(drop=True)
    )

    n = len(patient_times)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)

    patient_times["temporal_split"] = "test"
    patient_times.loc[: train_end - 1, "temporal_split"] = "train"
    patient_times.loc[train_end: val_end - 1, "temporal_split"] = "validate"

    out = df.merge(
        patient_times[["subject_id", "patient_first_t0", "temporal_split"]],
        on="subject_id",
        how="left",
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    report = {
        "rows": int(len(out)),
        "subjects": int(out["subject_id"].nunique()),
        "studies": int(out["study_id"].nunique()),
        "split_counts_rows": {
            str(k): int(v)
            for k, v in out["temporal_split"].value_counts(dropna=False).to_dict().items()
        },
        "split_counts_subjects": {
            str(k): int(v)
            for k, v in patient_times["temporal_split"].value_counts(dropna=False).to_dict().items()
        },
        "date_range": {
            "min_t0": str(out["t0"].min()),
            "max_t0": str(out["t0"].max()),
        },
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()