from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def find_single_file(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return matches[0]


def load_core_tables(metadata_root: Path):
    metadata_path = find_single_file(metadata_root, "*metadata.csv.gz")
    split_path = find_single_file(metadata_root, "*split.csv.gz")

    metadata = pd.read_csv(metadata_path)
    split_df = pd.read_csv(split_path)

    return metadata, split_df


def make_expected_image_path(base_root: Path, subject_id: int, study_id: int, dicom_id: str) -> Path:
    subject_str = str(subject_id)
    prefix2 = subject_str[:2]

    return (
        base_root
        / f"files_p{prefix2}"
        / "mimic-cxr-jpg"
        / "2.1.0"
        / "files"
        / f"p{prefix2}"
        / f"p{subject_str}"
        / f"s{study_id}"
        / f"{dicom_id}.jpg"
    )


def build_t0(df: pd.DataFrame) -> pd.Series:
    study_date = df["StudyDate"].astype("Int64").astype(str).str.strip()

    # Keep only the integer HHMMSS part before any decimal point,
    # then left-pad to 6 digits so e.g. 44509 -> 044509
    study_time = (
        df["StudyTime"]
        .astype(str)
        .str.strip()
        .str.split(".")
        .str[0]
        .replace({"": None, "nan": None, "NaT": None, "<NA>": None})
    )

    study_time = study_time.fillna("000000").str.zfill(6)

    hh = study_time.str.slice(0, 2)
    mm = study_time.str.slice(2, 4)
    ss = study_time.str.slice(4, 6)

    # Clamp obviously invalid components to missing instead of silently producing garbage
    valid = (
        hh.astype(int).between(0, 23)
        & mm.astype(int).between(0, 59)
        & ss.astype(int).between(0, 59)
    )

    dt_str = study_date + hh + mm + ss
    t0 = pd.to_datetime(dt_str.where(valid), format="%Y%m%d%H%M%S", errors="coerce")

    return t0


def add_view_flags(df: pd.DataFrame) -> pd.DataFrame:
    view = df["ViewPosition"].fillna("").astype(str).str.upper().str.strip()

    df["is_pa"] = view.eq("PA")
    df["is_ap"] = view.eq("AP")
    df["is_lateral"] = view.isin(["LATERAL", "LL", "LAT"])
    df["is_frontal"] = df["is_pa"] | df["is_ap"]

    return df


def add_study_image_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df.groupby(["subject_id", "study_id"])["dicom_id"]
        .count()
        .rename("n_images_in_study")
        .reset_index()
    )

    df = df.merge(counts, on=["subject_id", "study_id"], how="left")
    df["has_single_image"] = df["n_images_in_study"].eq(1)
    df["has_multiple_images"] = df["n_images_in_study"].gt(1)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=str, required=True)
    parser.add_argument("--metadata-root", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument(
        "--output-manifest",
        type=str,
        default="artifacts/manifests/mimic_cxr_manifest.parquet",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="artifacts/manifests/mimic_cxr_manifest_report.json",
    )
    args = parser.parse_args()

    base_root = Path(args.base_root)
    metadata_root = Path(args.metadata_root)

    metadata, split_df = load_core_tables(metadata_root)
    df = metadata.merge(split_df, on=["subject_id", "study_id", "dicom_id"], how="left")

    if args.limit:
        df = df.head(args.limit).copy()

    df["image_path"] = df.apply(
        lambda row: str(
            make_expected_image_path(
                base_root,
                int(row["subject_id"]),
                int(row["study_id"]),
                str(row["dicom_id"]),
            )
        ),
        axis=1,
    )

    if args.verify_paths:
        df["exists"] = df["image_path"].map(lambda p: Path(p).exists())
    else:
        df["exists"] = pd.NA

    df["t0"] = build_t0(df)
    df = add_view_flags(df)
    df = add_study_image_counts(df)

    output_cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "split",
        "StudyDate",
        "StudyTime",
        "t0",
        "ViewPosition",
        "is_pa",
        "is_ap",
        "is_lateral",
        "is_frontal",
        "n_images_in_study",
        "has_single_image",
        "has_multiple_images",
        "image_path",
        "exists",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    manifest = df[output_cols].copy()

    output_path = Path(args.output_manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False)

    report = {
        "rows": int(len(manifest)),
        "subjects": int(manifest["subject_id"].nunique()),
        "studies": int(manifest["study_id"].nunique()),
        "frontal_rows": int(manifest["is_frontal"].sum()),
        "pa_rows": int(manifest["is_pa"].sum()),
        "ap_rows": int(manifest["is_ap"].sum()),
        "lateral_rows": int(manifest["is_lateral"].sum()),
        "single_image_rows": int(manifest["has_single_image"].sum()),
        "multi_image_rows": int(manifest["has_multiple_images"].sum()),
        "missing_t0": int(manifest["t0"].isna().sum()),
    }

    if args.verify_paths:
        report["existing"] = int(manifest["exists"].fillna(False).sum())
        report["missing"] = int((~manifest["exists"].fillna(False)).sum())

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()