from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_cxr(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.copy()


def load_admissions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])

    return df


def link_cxr_to_admissions(cxr: pd.DataFrame, adm: pd.DataFrame) -> pd.DataFrame:
    # Merge on subject_id
    merged = cxr.merge(adm, on="subject_id", how="left")

    # Keep only rows where t0 is within admission window
    in_window = (
        (merged["t0"] >= merged["admittime"]) &
        (merged["t0"] <= merged["dischtime"])
    )

    linked = merged[in_window].copy()

    return linked


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cxr-cohort",
        type=str,
        default="artifacts/manifests/mimic_cxr_primary_frontal_cohort.parquet",
    )

    parser.add_argument(
        "--admissions",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/cxr_admissions_linked.parquet",
    )

    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/cxr_admissions_linked_report.json",
    )

    args = parser.parse_args()

    cxr = load_cxr(Path(args.cxr_cohort))
    adm = load_admissions(Path(args.admissions))

    linked = link_cxr_to_admissions(cxr, adm)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    linked.to_parquet(args.output, index=False)

    report = {
        "input_cxr_rows": int(len(cxr)),
        "linked_rows": int(len(linked)),
        "linked_subjects": int(linked["subject_id"].nunique()),
        "linked_studies": int(linked["study_id"].nunique()),
        "admission_ids": int(linked["hadm_id"].nunique()),
    }

    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()