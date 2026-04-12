from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_cxr(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


def load_triage(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Parse datetime columns if present
    for col in ["intime", "outtime", "charttime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cxr-cohort",
        type=str,
        default="artifacts/manifests/cxr_final_ed_cohort.parquet",
    )

    parser.add_argument(
        "--triage",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/cxr_ed_triage_linked.parquet",
    )

    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/cxr_ed_triage_linked_report.json",
    )

    args = parser.parse_args()

    cxr = load_cxr(Path(args.cxr_cohort))
    triage = load_triage(Path(args.triage))

    id_col = "stay_id" if "stay_id" in cxr.columns else "ed_stay_id"

    # Temporal safety note: MIMIC-IV-ED triage is a per-stay intake summary.
    # Triage assessment structurally precedes CXR ordering in ED workflow.
    # No charttime filter needed — the data is a single intake snapshot.
    merged = cxr.merge(triage, on=id_col, how="left")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)

    report = {
        "input_rows": int(len(cxr)),
        "linked_rows": int(len(merged)),
        "missing_triage_rows": int(merged[id_col].isna().sum()),
        "columns": list(merged.columns),
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()