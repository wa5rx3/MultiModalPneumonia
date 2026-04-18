from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


TARGET_KEYWORDS = [

    "wbc", "white blood",
    "hemoglobin", "hematocrit",
    "platelet",


    "sodium", "potassium", "chloride",
    "bicarbonate", "co2",
    "urea", "bun",
    "creatinine",
    "glucose",
    "calcium",


    "ph",
    "pco2",
    "po2",
    "base excess",


    "lactate",
    "crp",
    "procalcitonin",


    "ast", "alt",
    "bilirubin",
    "alkaline phosphatase",


    "albumin",
    "total protein",


    "anion gap"
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--d-labitems",
        type=str,
        default=None,
        help="Path to d_labitems.csv.gz from MIMIC-IV (required)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/tables/lab_feature_candidates.csv",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="artifacts/logs/lab_feature_candidates_report.json",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.d_labitems).copy()


    for col in ["label", "fluid", "category"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    text = pd.Series("", index=df.index, dtype="string")
    for col in ["label", "fluid", "category"]:
        if col in df.columns:
            text = text.fillna("") + " " + df[col].fillna("")

    text = text.str.lower()

    mask = pd.Series(False, index=df.index)
    for kw in TARGET_KEYWORDS:
        mask |= text.str.contains(kw, regex=False, na=False)

    out = df[mask].copy()

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(args.output_csv, index=False)

    report = {
        "rows_in_dictionary": int(len(df)),
        "candidate_rows": int(len(out)),
        "unique_itemids": int(out["itemid"].nunique()) if "itemid" in out.columns else None,
        "keywords": TARGET_KEYWORDS,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()