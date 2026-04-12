from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def find_single_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return matches[0]


def load_chexpert(metadata_root: Path) -> pd.DataFrame:
    chexpert_path = find_single_file(metadata_root, "*chexpert.csv.gz")
    df = pd.read_csv(chexpert_path).copy()
    return df


def get_merge_keys(cohort: pd.DataFrame, chex: pd.DataFrame, allow_fallback: bool) -> list[str]:
    preferred = ["subject_id", "study_id", "dicom_id"]
    fallback = ["subject_id", "study_id"]

    preferred_ok = all(col in cohort.columns for col in preferred) and all(col in chex.columns for col in preferred)
    fallback_ok = all(col in cohort.columns for col in fallback) and all(col in chex.columns for col in fallback)

    if preferred_ok:
        return preferred

    if allow_fallback and fallback_ok:
        return fallback

    raise ValueError(
        "Could not find acceptable merge keys.\n"
        f"Preferred keys required: {preferred}\n"
        f"Allow fallback: {allow_fallback}\n"
        f"Cohort columns: {list(cohort.columns)}\n"
        f"CheXpert columns: {list(chex.columns)}"
    )


def build_conflict_report(chex: pd.DataFrame, group_keys: list[str]) -> dict:
    # Only keep rows with a non-null pneumonia label for conflict analysis
    sub = chex[group_keys + ["Pneumonia"]].copy()
    sub = sub.dropna(subset=["Pneumonia"])

    if sub.empty:
        return {
            "group_keys": group_keys,
            "groups_total": 0,
            "groups_with_duplicate_rows": 0,
            "groups_with_conflicting_pneumonia_labels": 0,
        }

    grp = (
        sub.groupby(group_keys, dropna=False)["Pneumonia"]
        .agg(
            n_rows="size",
            n_unique_labels="nunique",
        )
        .reset_index()
    )

    return {
        "group_keys": group_keys,
        "groups_total": int(len(grp)),
        "groups_with_duplicate_rows": int((grp["n_rows"] > 1).sum()),
        "groups_with_conflicting_pneumonia_labels": int((grp["n_unique_labels"] > 1).sum()),
    }


def collapse_chexpert(chex: pd.DataFrame, merge_keys: list[str]) -> tuple[pd.DataFrame, dict]:
    needed_cols = [c for c in merge_keys + ["Pneumonia"] if c in chex.columns]
    sub = chex[needed_cols].copy()

    # QC before collapsing
    pre_qc = build_conflict_report(sub, merge_keys)

    # If there are conflicting labels at the merge-key level, fail loudly.
    if pre_qc["groups_with_conflicting_pneumonia_labels"] > 0:
        raise ValueError(
            "Conflicting CheXpert pneumonia labels found for the selected merge keys. "
            f"QC: {pre_qc}"
        )

    # Safe collapse after conflict check
    sub = sub.drop_duplicates()

    # After duplicates removed, there should be at most one row per merge key.
    key_counts = sub.groupby(merge_keys, dropna=False).size()
    multi_key_rows = int((key_counts > 1).sum())
    if multi_key_rows > 0:
        raise ValueError(
            "Multiple CheXpert rows remain for selected merge keys after drop_duplicates. "
            f"multi_key_rows={multi_key_rows}"
        )

    post_qc = {
        "rows_after_drop_duplicates": int(len(sub)),
        "unique_merge_key_rows": int(len(key_counts)),
    }

    return sub, {"pre_qc": pre_qc, "post_qc": post_qc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cxr-cohort",
        type=str,
        default="artifacts/manifests/cxr_final_ed_cohort.parquet",
    )
    parser.add_argument(
        "--metadata-root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/cxr_pneumonia_labels.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/cxr_pneumonia_labels_report.json",
    )
    parser.add_argument(
        "--allow-fallback-study-merge",
        action="store_true",
        help="Allow fallback from (subject_id, study_id, dicom_id) to (subject_id, study_id). "
             "Use only if dicom_id is unavailable in CheXpert and document this clearly.",
    )
    args = parser.parse_args()

    cohort = pd.read_parquet(args.cxr_cohort).copy()
    chex = load_chexpert(Path(args.metadata_root)).copy()

    merge_keys = get_merge_keys(
        cohort=cohort,
        chex=chex,
        allow_fallback=args.allow_fallback_study_merge,
    )

    chex_collapsed, chex_qc = collapse_chexpert(chex, merge_keys)

    merged = cohort.merge(
        chex_collapsed,
        on=merge_keys,
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    merged["chexpert_row_found"] = merged["_merge"].eq("both")
    merged["pneumonia_chexpert_raw"] = merged["Pneumonia"]

    merged["pneumonia_positive"] = merged["pneumonia_chexpert_raw"].eq(1)
    merged["pneumonia_negative"] = merged["pneumonia_chexpert_raw"].eq(0)
    merged["pneumonia_uncertain"] = merged["pneumonia_chexpert_raw"].eq(-1)
    merged["pneumonia_missing"] = merged["pneumonia_chexpert_raw"].isna()

    output_cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "hadm_id",
        "stay_id",
        "split",
        "t0",
        "image_path",
        "chexpert_row_found",
        "pneumonia_chexpert_raw",
        "pneumonia_positive",
        "pneumonia_negative",
        "pneumonia_uncertain",
        "pneumonia_missing",
    ]
    output_cols = [c for c in output_cols if c in merged.columns]

    out = merged[output_cols].copy()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    report = {
        "rows": int(len(out)),
        "merge_keys_used": merge_keys,
        "fallback_study_merge_allowed": bool(args.allow_fallback_study_merge),
        "chexpert_row_found": int(out["chexpert_row_found"].sum()),
        "chexpert_row_missing": int((~out["chexpert_row_found"]).sum()),
        "positive": int(out["pneumonia_positive"].sum()),
        "negative": int(out["pneumonia_negative"].sum()),
        "uncertain": int(out["pneumonia_uncertain"].sum()),
        "missing": int(out["pneumonia_missing"].sum()),
        "chexpert_qc": chex_qc,
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()