from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--linked-ed",
        type=str,
        default="artifacts/manifests/cxr_edstays_linked.parquet",
    )
    parser.add_argument(
        "--output-cohort",
        type=str,
        default="artifacts/manifests/cxr_final_ed_cohort.parquet",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="artifacts/manifests/cxr_final_ed_cohort_report.json",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.linked_ed).copy()

    study_key = ["subject_id", "study_id"]
    id_col = "stay_id" if "stay_id" in df.columns else "ed_stay_id"

    match_counts = (
        df.groupby(study_key)[id_col]
        .nunique()
        .rename("n_ed_matches")
        .reset_index()
    )

    valid = match_counts[match_counts["n_ed_matches"] == 1][study_key].copy()

    cohort = df.merge(valid, on=study_key, how="inner").copy()


    cohort = cohort.sort_values(study_key + [id_col]).groupby(study_key, as_index=False).head(1).copy()

    output_path = Path(args.output_cohort)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(output_path, index=False)

    report = {
        "final_rows": int(len(cohort)),
        "final_subjects": int(cohort["subject_id"].nunique()),
        "final_studies": int(cohort["study_id"].nunique()),
        "final_ed_stays": int(cohort[id_col].nunique()),
        "missing_t0": int(cohort["t0"].isna().sum()) if "t0" in cohort.columns else None,
    }

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()