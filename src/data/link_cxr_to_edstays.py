from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_cxr(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


def load_edstays(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["intime"] = pd.to_datetime(df["intime"])
    df["outtime"] = pd.to_datetime(df["outtime"])
    return df


def link_cxr_to_edstays(cxr: pd.DataFrame, ed: pd.DataFrame) -> pd.DataFrame:
    merged = cxr.merge(ed, on="subject_id", how="left")

    in_window = (
        (merged["t0"] >= merged["intime"]) &
        (merged["t0"] <= merged["outtime"])
    )

    linked = merged[in_window].copy()
    return linked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cxr-cohort",
        type=str,
        default="artifacts/manifests/mimic_cxr_primary_frontal_cohort.parquet",
    )
    parser.add_argument(
        "--edstays",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/cxr_edstays_linked.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/cxr_edstays_linked_report.json",
    )
    args = parser.parse_args()

    cxr = load_cxr(Path(args.cxr_cohort))
    ed = load_edstays(Path(args.edstays))

    linked = link_cxr_to_edstays(cxr, ed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    linked.to_parquet(args.output, index=False)

    report = {
        "input_cxr_rows": int(len(cxr)),
        "linked_rows": int(len(linked)),
        "linked_subjects": int(linked["subject_id"].nunique()),
        "linked_studies": int(linked["study_id"].nunique()),
        "ed_stay_ids": int(linked["stay_id"].nunique()) if "stay_id" in linked.columns else None,
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()