from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_feature_map(path: str) -> dict[str, list[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: [int(v) for v in vals] for k, vals in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-labs",
        type=str,
        default="artifacts/tables/cohort_labevents.parquet",
    )
    parser.add_argument(
        "--feature-map",
        type=str,
        default="artifacts/tables/lab_feature_map.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/tables/cxr_lab_features.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/logs/cxr_lab_features_report.json",
    )
    args = parser.parse_args()

    labs = pd.read_parquet(args.input_labs).copy()
    feature_map = load_feature_map(args.feature_map)

    needed = ["subject_id", "study_id", "itemid", "charttime", "valuenum"]
    missing = [c for c in needed if c not in labs.columns]
    if missing:
        raise ValueError(f"Input lab table missing required columns: {missing}")

    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    labs["valuenum"] = pd.to_numeric(labs["valuenum"], errors="coerce")
    labs = labs.dropna(subset=["charttime", "valuenum"])

    # Build reverse map: itemid -> concept
    itemid_to_concept: dict[int, str] = {}
    for concept, itemids in feature_map.items():
        for itemid in itemids:
            itemid_to_concept[itemid] = concept

    labs["lab_concept"] = labs["itemid"].map(itemid_to_concept)
    labs = labs.dropna(subset=["lab_concept"])

    # Sort so "last value before t0" is the last row per study/concept
    labs = labs.sort_values(["subject_id", "study_id", "lab_concept", "charttime"])

    last_vals = (
        labs.groupby(["subject_id", "study_id", "lab_concept"], as_index=False)
        .tail(1)
        .copy()
    )

    feat = last_vals.pivot_table(
        index=["subject_id", "study_id"],
        columns="lab_concept",
        values="valuenum",
        aggfunc="first",
    ).reset_index()

    # Flatten pivoted column names
    feat.columns.name = None

    # Add missingness flags
    concept_cols = [c for c in feat.columns if c not in ["subject_id", "study_id"]]
    for col in concept_cols:
        feat[f"{col}_missing"] = feat[col].isna()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(args.output, index=False)

    report = {
        "rows": int(len(feat)),
        "subjects": int(feat["subject_id"].nunique()),
        "studies": int(feat["study_id"].nunique()),
        "lab_feature_columns": concept_cols,
        "num_lab_features": int(len(concept_cols)),
        "missing_counts": {
            col: int(feat[col].isna().sum())
            for col in concept_cols
        },
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()