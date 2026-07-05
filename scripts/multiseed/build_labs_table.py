"""Build a time-safe triage+labs training table on the rebuilt u_ignore cohort.

Left-joins the rebuilt clinical (triage) temporal table with the lab-feature table
(last value at or before t0, 24h lookback, subject+encounter matched). Missingness
flags are recomputed AFTER the join so studies with no labs are correctly flagged
missing rather than NaN. The sacred temporal split from the triage table is kept
verbatim, so image-only / triage-multimodal / labs-multimodal all evaluate on the
identical held-out test set.

Outputs artifacts/manifests/cxr_clinical_labs_training_table_u_ignore_temporal.parquet
and a coverage report.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

TRIAGE = Path("artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet")
LAB_FEATURES = Path("artifacts/tables/cxr_lab_features.parquet")
FEATURE_MAP = Path("artifacts/tables/lab_feature_map.json")
OUT = Path("artifacts/manifests/cxr_clinical_labs_training_table_u_ignore_temporal.parquet")
REPORT = Path("artifacts/logs/cxr_clinical_labs_training_table_u_ignore_temporal_report.json")


def lab_concepts() -> list[str]:
    with open(FEATURE_MAP, "r", encoding="utf-8") as f:
        return list(json.load(f).keys())


def main() -> None:
    tri = pd.read_parquet(TRIAGE)
    labs = pd.read_parquet(LAB_FEATURES)

    concepts = lab_concepts()
    value_cols = [c for c in concepts if c in labs.columns]
    if not value_cols:
        raise SystemExit(f"No lab concept columns found in {LAB_FEATURES}")

    # keep one row per (subject, study); take lab values only (recompute missingness post-join)
    labs = labs[["subject_id", "study_id"] + value_cols].drop_duplicates(["subject_id", "study_id"])

    merged = tri.merge(labs, on=["subject_id", "study_id"], how="left")
    for c in value_cols:
        merged[f"{c}_missing"] = merged[c].isna().astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False)

    # coverage report per split
    report = {
        "output": str(OUT),
        "rows": int(len(merged)),
        "lab_value_columns": value_cols,
        "num_lab_value_columns": len(value_cols),
        "split_counts": {s: int((merged["temporal_split"] == s).sum())
                         for s in ["train", "validate", "test"]},
        "any_lab_present_by_split": {},
        "per_lab_coverage_pct_test": {},
    }
    for s in ["train", "validate", "test"]:
        sub = merged[merged["temporal_split"] == s]
        any_present = (sub[value_cols].notna().any(axis=1)).mean() if len(sub) else 0.0
        report["any_lab_present_by_split"][s] = round(float(any_present), 4)
    test = merged[merged["temporal_split"] == "test"]
    for c in value_cols:
        report["per_lab_coverage_pct_test"][c] = round(float(test[c].notna().mean() * 100), 1)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: report[k] for k in
                      ["rows", "num_lab_value_columns", "split_counts", "any_lab_present_by_split"]}, indent=2))
    print("\nTest-set per-lab coverage (%):")
    for c, v in sorted(report["per_lab_coverage_pct_test"].items(), key=lambda x: -x[1]):
        print(f"  {c:20s} {v:5.1f}%")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
