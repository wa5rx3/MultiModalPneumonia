from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cohort", type=str, required=True)
    parser.add_argument("--input-table", type=str, required=True)
    parser.add_argument("--output-table", type=str, required=True)
    parser.add_argument("--report", type=str, required=True)
    args = parser.parse_args()

    cohort = pd.read_parquet(args.base_cohort).copy()
    table = pd.read_parquet(args.input_table).copy()

    split_cols = ["subject_id", "study_id", "temporal_split"]
    split_map = cohort[split_cols].drop_duplicates()


    if "split" in table.columns:
        table = table.drop(columns=["split"])

    merged = table.merge(split_map, on=["subject_id", "study_id"], how="left")

    report = {
        "input_rows": int(len(table)),
        "output_rows": int(len(merged)),
        "missing_temporal_split": int(merged["temporal_split"].isna().sum()),
        "split_counts": {
            str(k): int(v)
            for k, v in merged["temporal_split"].value_counts(dropna=False).to_dict().items()
        },
    }

    Path(args.output_table).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output_table, index=False)

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()