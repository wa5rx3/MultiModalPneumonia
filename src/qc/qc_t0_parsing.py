from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        type=str,
        default="artifacts/manifests/mimic_cxr_manifest_sample.parquet",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="artifacts/manifests/mimic_cxr_missing_t0_rows.csv",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input_manifest).copy()

    missing = df[df["t0"].isna()].copy()

    cols = [
        "subject_id",
        "study_id",
        "dicom_id",
        "StudyDate",
        "StudyTime",
        "ViewPosition",
        "image_path",
    ]
    cols = [c for c in cols if c in missing.columns]
    missing = missing[cols]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(out_path, index=False)

    print(f"missing_t0_rows={len(missing)}")
    if len(missing) > 0:
        print(missing.head(20).to_string(index=False))


if __name__ == "__main__":
    main()