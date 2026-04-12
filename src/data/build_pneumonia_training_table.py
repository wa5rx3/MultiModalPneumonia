from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=str,
        default="artifacts/manifests/cxr_pneumonia_labels.parquet",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["u_ignore", "u_zero", "u_one"],
        default="u_ignore",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/manifests/cxr_pneumonia_training_table.parquet",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="artifacts/manifests/cxr_pneumonia_training_table_report.json",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.labels).copy()

    if args.policy == "u_ignore":
        keep = df["pneumonia_positive"] | df["pneumonia_negative"]
        out = df[keep].copy()
        out["target"] = out["pneumonia_positive"].astype(int)

    elif args.policy == "u_zero":
        keep = df["pneumonia_positive"] | df["pneumonia_negative"] | df["pneumonia_uncertain"]
        out = df[keep].copy()
        out["target"] = out["pneumonia_positive"].astype(int)

    elif args.policy == "u_one":
        keep = df["pneumonia_positive"] | df["pneumonia_negative"] | df["pneumonia_uncertain"]
        out = df[keep].copy()
        out["target"] = (out["pneumonia_positive"] | out["pneumonia_uncertain"]).astype(int)

    else:
        raise ValueError(f"Unsupported policy: {args.policy}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)

    report = {
        "policy": args.policy,
        "rows": int(len(out)),
        "positives": int((out["target"] == 1).sum()),
        "negatives": int((out["target"] == 0).sum()),
        "positive_rate": float(out["target"].mean()) if len(out) > 0 else None,
        "split_counts": {
            str(k): int(v) for k, v in out["split"].value_counts(dropna=False).to_dict().items()
        } if "split" in out.columns else {},
    }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()