from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cxr-cohort",
        type=str,
        default="artifacts/manifests/mimic_cxr_primary_frontal_cohort.parquet",
    )
    parser.add_argument(
        "--linked",
        type=str,
        default="artifacts/manifests/cxr_admissions_linked.parquet",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="artifacts/manifests/cxr_admissions_linkage_qc_report.json",
    )
    parser.add_argument(
        "--multi-match-csv",
        type=str,
        default="artifacts/manifests/cxr_admissions_multi_match.csv",
    )
    args = parser.parse_args()

    cxr = pd.read_parquet(args.cxr_cohort)
    linked = pd.read_parquet(args.linked)

    study_key = ["subject_id", "study_id"]

    total_input_studies = cxr[study_key].drop_duplicates().shape[0]
    linked_studies_df = linked[study_key].drop_duplicates()
    linked_studies = linked_studies_df.shape[0]
    unlinked_studies = total_input_studies - linked_studies

    matches_per_study = (
        linked.groupby(study_key)["hadm_id"]
        .nunique()
        .rename("n_hadm_matches")
        .reset_index()
    )

    one_match = int((matches_per_study["n_hadm_matches"] == 1).sum())
    multi_match = int((matches_per_study["n_hadm_matches"] > 1).sum())

    multi_match_df = matches_per_study[matches_per_study["n_hadm_matches"] > 1].copy()
    if not multi_match_df.empty:
        multi_detail = linked.merge(multi_match_df[study_key], on=study_key, how="inner").copy()
    else:
        multi_detail = linked.head(0).copy()

    admissions_with_multiple_cxrs = (
        linked.groupby("hadm_id")["study_id"]
        .nunique()
        .gt(1)
        .sum()
    )

    report = {
        "input_studies": int(total_input_studies),
        "linked_studies": int(linked_studies),
        "unlinked_studies": int(unlinked_studies),
        "studies_with_exactly_one_hadm_match": one_match,
        "studies_with_multiple_hadm_matches": multi_match,
        "unique_hadm_ids": int(linked["hadm_id"].nunique()),
        "admissions_with_multiple_linked_cxrs": int(admissions_with_multiple_cxrs),
    }

    Path(args.output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    Path(args.multi_match_csv).parent.mkdir(parents=True, exist_ok=True)
    multi_detail.to_csv(args.multi_match_csv, index=False)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()