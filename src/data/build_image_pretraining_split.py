from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ALLOWED_TEMPORAL_SPLITS = {"train", "validate", "test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary-frontal-cohort",
        type=str,
        default="artifacts/manifests/mimic_cxr_primary_frontal_cohort.parquet",
    )
    parser.add_argument(
        "--ed-temporal-cohort",
        type=str,
        default="artifacts/manifests/cxr_final_ed_cohort_with_temporal_split.parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/mimic_cxr_primary_frontal_with_pretrain_split.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/mimic_cxr_primary_frontal_with_pretrain_split_report.json",
    )
    parser.add_argument(
        "--internal-val-frac",
        type=float,
        default=0.1,
        help="Fraction of non-ED subjects reserved for pretraining internal validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="allow_ed_train",
        choices=["allow_ed_train", "exclude_all_ed"],
        help=(
            "allow_ed_train: ED train subjects may be used in supervised pretraining train. "
            "exclude_all_ed: all ED subjects are excluded from supervised pretraining."
        ),
    )
    args = parser.parse_args()

    if not (0.0 < args.internal_val_frac < 1.0):
        raise ValueError("--internal-val-frac must be between 0 and 1.")

    print("Loading primary frontal cohort...")
    primary = pd.read_parquet(args.primary_frontal_cohort).copy()

    print("Loading ED temporal cohort...")
    ed = pd.read_parquet(args.ed_temporal_cohort).copy()

    required_primary = ["subject_id", "study_id", "dicom_id", "image_path"]
    missing_primary = [c for c in required_primary if c not in primary.columns]
    if missing_primary:
        raise ValueError(f"Primary frontal cohort missing columns: {missing_primary}")

    required_ed = ["subject_id", "temporal_split"]
    missing_ed = [c for c in required_ed if c not in ed.columns]
    if missing_ed:
        raise ValueError(f"ED temporal cohort missing columns: {missing_ed}")

    primary["subject_id"] = pd.to_numeric(primary["subject_id"], errors="raise").astype("int64")
    ed["subject_id"] = pd.to_numeric(ed["subject_id"], errors="raise").astype("int64")

    split_values = set(ed["temporal_split"].dropna().unique().tolist())
    bad_split_values = split_values - ALLOWED_TEMPORAL_SPLITS
    if bad_split_values:
        raise ValueError(
            f"Unexpected temporal_split values found: {sorted(bad_split_values)}. "
            f"Allowed: {sorted(ALLOWED_TEMPORAL_SPLITS)}"
        )

    ed_subject_split = ed[["subject_id", "temporal_split"]].drop_duplicates().copy()

    dup_subjects = (
        ed_subject_split.groupby("subject_id")["temporal_split"].nunique().reset_index(name="n")
    )
    bad_subjects = int((dup_subjects["n"] > 1).sum())
    if bad_subjects > 0:
        raise ValueError(
            f"Found {bad_subjects} ED subjects assigned to multiple temporal splits."
        )

    ed_subject_split = ed_subject_split.drop_duplicates(subset=["subject_id"])

    ed_train_subjects = set(
        ed_subject_split.loc[ed_subject_split["temporal_split"] == "train", "subject_id"].tolist()
    )
    ed_validate_subjects = set(
        ed_subject_split.loc[ed_subject_split["temporal_split"] == "validate", "subject_id"].tolist()
    )
    ed_test_subjects = set(
        ed_subject_split.loc[ed_subject_split["temporal_split"] == "test", "subject_id"].tolist()
    )
    ed_all_subjects = set(ed_subject_split["subject_id"].tolist())

    primary_subjects = set(primary["subject_id"].unique().tolist())

    if args.policy == "allow_ed_train":
        eligible_non_ed_subjects = sorted(primary_subjects - ed_all_subjects)
    elif args.policy == "exclude_all_ed":
        eligible_non_ed_subjects = sorted(primary_subjects - ed_all_subjects)
    else:
        raise ValueError(f"Unsupported policy: {args.policy}")

    rng = np.random.default_rng(args.seed)
    eligible_non_ed_subjects = np.array(eligible_non_ed_subjects, dtype=np.int64)
    rng.shuffle(eligible_non_ed_subjects)

    n_non_ed = len(eligible_non_ed_subjects)
    n_internal_val = math.floor(n_non_ed * args.internal_val_frac)
    if n_non_ed > 0:
        n_internal_val = max(1, n_internal_val)
        n_internal_val = min(n_internal_val, n_non_ed)

    non_ed_internal_val_subjects = set(eligible_non_ed_subjects[:n_internal_val].tolist())
    non_ed_train_subjects = set(eligible_non_ed_subjects[n_internal_val:].tolist())

    def assign_pretrain_split(subject_id: int) -> str:
        if subject_id in ed_validate_subjects:
            return "exclude_ed_validate"
        if subject_id in ed_test_subjects:
            return "exclude_ed_test"
        if args.policy == "exclude_all_ed" and subject_id in ed_train_subjects:
            return "exclude_ed_train"
        if args.policy == "allow_ed_train" and subject_id in ed_train_subjects:
            return "pretrain_train"
        if subject_id in non_ed_internal_val_subjects:
            return "pretrain_internal_val"
        if subject_id in non_ed_train_subjects:
            return "pretrain_train"
        raise ValueError(f"Unassigned subject_id: {subject_id}")

    primary["pretrain_split"] = primary["subject_id"].map(assign_pretrain_split)

    primary["is_in_ed_temporal_cohort"] = primary["subject_id"].isin(ed_all_subjects)
    primary["is_ed_train_subject"] = primary["subject_id"].isin(ed_train_subjects)
    primary["is_ed_validate_subject"] = primary["subject_id"].isin(ed_validate_subjects)
    primary["is_ed_test_subject"] = primary["subject_id"].isin(ed_test_subjects)
    primary["exclude_from_supervised_pretraining"] = primary["pretrain_split"].str.startswith("exclude_")


    primary_subjects_set = set(primary["subject_id"].unique().tolist())
    required_excluded = (ed_validate_subjects | ed_test_subjects) & primary_subjects_set
    actually_excluded = set(
        primary.loc[primary["exclude_from_supervised_pretraining"], "subject_id"].unique().tolist()
    )
    missing_required_exclusions = required_excluded - actually_excluded
    if missing_required_exclusions:
        raise ValueError(
            f"Leakage risk: {len(missing_required_exclusions)} ED validate/test subjects "
            "present in primary are not excluded from supervised pretraining."
        )


    if args.policy == "exclude_all_ed":
        required_excluded_all_ed = ed_all_subjects & primary_subjects_set
        missing_all_ed_exclusions = required_excluded_all_ed - actually_excluded
        if missing_all_ed_exclusions:
            raise ValueError(
                f"Leakage risk: {len(missing_all_ed_exclusions)} ED subjects present in primary "
                "are not excluded under exclude_all_ed policy."
            )


    if args.policy == "allow_ed_train":
        allowed_excluded_subjects = required_excluded
    else:
        allowed_excluded_subjects = ed_all_subjects & primary_subjects_set

    unexpected_excluded_subjects = actually_excluded - allowed_excluded_subjects
    if unexpected_excluded_subjects:
        raise ValueError(
            f"Found {len(unexpected_excluded_subjects)} excluded subjects outside intended ED exclusion set."
        )


    trainable_subjects_seen = set(
        primary.loc[
            primary["pretrain_split"].isin(["pretrain_train", "pretrain_internal_val"]),
            "subject_id",
        ].unique().tolist()
    )
    forbidden_train_subjects = trainable_subjects_seen & (ed_validate_subjects | ed_test_subjects)
    if forbidden_train_subjects:
        raise ValueError(
            f"Leakage: found {len(forbidden_train_subjects)} ED validate/test subjects in trainable pretraining pool."
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    primary.to_parquet(output_path, index=False)

    report = {
        "rows": int(len(primary)),
        "studies": int(primary["study_id"].nunique()),
        "subjects": int(primary["subject_id"].nunique()),
        "seed": int(args.seed),
        "policy": args.policy,
        "internal_val_frac": float(args.internal_val_frac),
        "ed_subject_counts": {
            "train": int(len(ed_train_subjects)),
            "validate": int(len(ed_validate_subjects)),
            "test": int(len(ed_test_subjects)),
            "total": int(len(ed_all_subjects)),
        },
        "non_ed_subject_counts": {
            "train": int(len(non_ed_train_subjects)),
            "internal_val": int(len(non_ed_internal_val_subjects)),
            "total": int(n_non_ed),
        },
        "pretrain_split_row_counts": {
            k: int(v)
            for k, v in primary["pretrain_split"].value_counts(dropna=False).to_dict().items()
        },
        "pretrain_split_subject_counts": {
            k: int(v)
            for k, v in primary.groupby("pretrain_split")["subject_id"].nunique().to_dict().items()
        },
        "excluded_from_pretraining_rows": int(primary["exclude_from_supervised_pretraining"].sum()),
        "excluded_from_pretraining_subjects": int(
            primary.loc[primary["exclude_from_supervised_pretraining"], "subject_id"].nunique()
        ),
        "notes": [
            "ED validate/test subjects are always excluded from supervised pretraining.",
            "Policy controls whether ED train subjects are allowed or excluded.",
            "Non-ED subjects are split by subject into pretrain_train and pretrain_internal_val.",
        ],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()