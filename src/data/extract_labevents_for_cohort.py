from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


LABEVENTS_COLUMNS = [
    "labevent_id",
    "subject_id",
    "hadm_id",
    "specimen_id",
    "itemid",
    "order_provider_id",
    "charttime",
    "storetime",
    "value",
    "valuenum",
    "valueuom",
    "ref_range_lower",
    "ref_range_upper",
    "flag",
    "priority",
    "comments",
]


def load_feature_map(path: str) -> set[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    itemids = set()
    for v in data.values():
        itemids.update(int(x) for x in v)
    return itemids


def coerce_hadm_id(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--labevents-dir", type=str, required=True)
    parser.add_argument("--cohort", type=str, required=True)
    parser.add_argument("--feature-map", type=str, required=True)

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--report", type=str, required=True)

    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--lookback-hours", type=int, default=24)

    parser.add_argument(
        "--match-mode",
        type=str,
        default="hadm_only",
        choices=["hadm_only", "hadm_plus_fallback"],
        help="hadm_only = strict encounter-safe. hadm_plus_fallback = adds subject-only fallback.",
    )

    args = parser.parse_args()

    print("Loading cohort...")
    cohort = pd.read_parquet(args.cohort)

    required = ["subject_id", "study_id", "hadm_id", "t0"]
    missing = [c for c in required if c not in cohort.columns]
    if missing:
        raise ValueError(f"Cohort missing columns: {missing}")

    cohort = cohort[required].copy()
    cohort["t0"] = pd.to_datetime(cohort["t0"], errors="coerce")
    cohort["hadm_id"] = coerce_hadm_id(cohort["hadm_id"])
    cohort = cohort.dropna(subset=["t0"])

    cohort_hadm = cohort[cohort["hadm_id"].notna()].copy()
    cohort_no_hadm = cohort[cohort["hadm_id"].isna()].copy()

    subject_ids = set(cohort["subject_id"].unique())

    print("Loading feature map...")
    target_itemids = load_feature_map(args.feature_map)

    lab_files = sorted(Path(args.labevents_dir).glob("*.csv.gz"))
    if not lab_files:
        raise FileNotFoundError("No labevents shards found")

    lookback = pd.Timedelta(hours=args.lookback_hours)

    print(f"Subjects: {len(subject_ids)}")
    print(f"Match mode: {args.match_mode}")
    print(f"Lab shards: {len(lab_files)}")

    outputs = []

    stats = {
        "rows_read": 0,
        "after_subject": 0,
        "after_itemid": 0,
        "after_clean": 0,
        "after_hadm_merge": 0,
        "after_fallback_merge": 0,
        "after_time_hadm": 0,
        "after_time_fallback": 0,
    }

    usecols = ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]

    for file in lab_files:
        print(f"\nProcessing {file.name}")

        for chunk in tqdm(
            pd.read_csv(
                file,
                header=None,
                names=LABEVENTS_COLUMNS,
                usecols=usecols,
                chunksize=args.chunksize,
                low_memory=False,
            ),
            desc=file.name,
        ):
            stats["rows_read"] += len(chunk)


            chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce").astype("Int64")
            chunk["hadm_id"] = coerce_hadm_id(chunk["hadm_id"])
            chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce").astype("Int64")
            chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
            chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")


            chunk = chunk[chunk["subject_id"].isin(subject_ids)]
            stats["after_subject"] += len(chunk)
            if chunk.empty:
                continue

            chunk = chunk[chunk["itemid"].isin(target_itemids)]
            stats["after_itemid"] += len(chunk)
            if chunk.empty:
                continue

            chunk = chunk.dropna(subset=["subject_id", "itemid", "charttime", "valuenum"])
            stats["after_clean"] += len(chunk)
            if chunk.empty:
                continue




            chunk_hadm = chunk[chunk["hadm_id"].notna()]
            if not chunk_hadm.empty:
                merged = chunk_hadm.merge(
                    cohort_hadm,
                    on=["subject_id", "hadm_id"],
                    how="inner",
                )
                stats["after_hadm_merge"] += len(merged)

                if not merged.empty:
                    merged = merged[
                        (merged["charttime"] <= merged["t0"]) &
                        (merged["charttime"] >= merged["t0"] - lookback)
                    ]
                    stats["after_time_hadm"] += len(merged)

                    if not merged.empty:
                        outputs.append(
                            merged[
                                ["subject_id", "study_id", "hadm_id", "itemid", "charttime", "valuenum"]
                            ]
                        )




            if args.match_mode == "hadm_plus_fallback":
                chunk_no_hadm = chunk[chunk["hadm_id"].isna()]
                if not chunk_no_hadm.empty and not cohort_no_hadm.empty:
                    merged = chunk_no_hadm.merge(
                        cohort_no_hadm,
                        on=["subject_id"],
                        how="inner",
                    )
                    stats["after_fallback_merge"] += len(merged)

                    if not merged.empty:
                        merged = merged[
                            (merged["charttime"] <= merged["t0"]) &
                            (merged["charttime"] >= merged["t0"] - lookback)
                        ]
                        stats["after_time_fallback"] += len(merged)

                        if not merged.empty:
                            merged["hadm_id"] = pd.NA
                            outputs.append(
                                merged[
                                    ["subject_id", "study_id", "hadm_id", "itemid", "charttime", "valuenum"]
                                ]
                            )

    if not outputs:
        raise RuntimeError("No lab rows extracted")

    print("\nFinalizing...")
    final = pd.concat(outputs, ignore_index=True)

    final = final.drop_duplicates().sort_values(
        ["subject_id", "study_id", "itemid", "charttime"]
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(args.output, index=False)

    report = {
        "match_mode": args.match_mode,
        "lookback_hours": args.lookback_hours,
        "num_files": len(lab_files),
        "final_rows": int(len(final)),
        "final_subjects": int(final["subject_id"].nunique()),
        "final_studies": int(final["study_id"].nunique()),
        "final_itemids": int(final["itemid"].nunique()),
        "stats": stats,
        "notes": [
            "Primary matching uses subject_id + hadm_id.",
            "Fallback only enabled when explicitly requested.",
            "Time window strictly before t0.",
        ],
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)

    print("\nDONE")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()