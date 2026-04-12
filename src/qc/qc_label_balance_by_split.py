from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--target-col", type=str, default="target")
    parser.add_argument("--split-col", type=str, default="temporal_split")
    parser.add_argument("--report", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input).copy()

    summary = {}
    for split_name, sub in df.groupby(args.split_col):
        pos = int((sub[args.target_col] == 1).sum())
        neg = int((sub[args.target_col] == 0).sum())
        summary[str(split_name)] = {
            "rows": int(len(sub)),
            "positives": pos,
            "negatives": neg,
            "positive_rate": float(sub[args.target_col].mean()) if len(sub) > 0 else None,
        }

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()