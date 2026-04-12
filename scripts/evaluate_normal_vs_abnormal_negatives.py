from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ABNORMAL_COLS = [
    "Atelectasis",
    "Edema",
    "Pleural Effusion",
    "Consolidation",
    "Lung Opacity",
]


def evaluate_subset(name: str, df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError(f"{name}: dataframe is empty")

    y_true = df["target"].astype(int).values
    y_prob = df["pred_prob"].astype(float).values

    if len(set(y_true)) < 2:
        raise ValueError(f"{name}: need both positive and negative classes")

    result = {
        "name": name,
        "n": int(len(df)),
        "positives": int((df["target"] == 1).sum()),
        "negatives": int((df["target"] == 0).sum()),
        "positive_rate": float(df["target"].mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", type=str, required=True)
    parser.add_argument("--chexpert-csv", type=str, required=True)
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save JSON summary",
    )
    parser.add_argument(
        "--merge-on-dicom",
        action="store_true",
        help="Force merge on subject_id, study_id, dicom_id. Use only if CheXpert CSV is unique per dicom.",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions_csv)
    chex_path = Path(args.chexpert_csv)

    if not pred_path.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {pred_path}")
    if not chex_path.is_file():
        raise FileNotFoundError(f"CheXpert CSV not found: {chex_path}")

    pred = pd.read_csv(pred_path).copy()
    chex = pd.read_csv(chex_path).copy()

    required_pred_cols = ["subject_id", "study_id", "target", "pred_prob"]
    missing_pred = [c for c in required_pred_cols if c not in pred.columns]
    if missing_pred:
        raise ValueError(f"Predictions CSV missing required columns: {missing_pred}")

    missing_abnormal = [c for c in ABNORMAL_COLS if c not in chex.columns]
    if missing_abnormal:
        raise ValueError(f"CheXpert CSV missing abnormality columns: {missing_abnormal}")

    # Decide merge keys
    can_merge_on_dicom = (
        args.merge_on_dicom
        and "dicom_id" in pred.columns
        and "dicom_id" in chex.columns
    )

    if can_merge_on_dicom:
        merge_keys = ["subject_id", "study_id", "dicom_id"]
    else:
        merge_keys = ["subject_id", "study_id"]

    # Collapse CheXpert to one row per merge key to avoid duplicate join explosions
    chex_small = chex[merge_keys + ABNORMAL_COLS].copy()
    chex_small = chex_small.drop_duplicates(subset=merge_keys)

    df = pred.merge(
        chex_small,
        on=merge_keys,
        how="left",
        validate="one_to_one",
    )

    initial_n = len(df)

    # Drop rows where all abnormality columns are NaN
    df = df.loc[~df[ABNORMAL_COLS].isna().all(axis=1)].copy()
    after_nan_drop_n = len(df)

    # Define abnormal: any explicit positive finding among selected abnormal cols
    df["any_abnormal"] = (df[ABNORMAL_COLS] == 1).any(axis=1)

    positives = df.loc[df["target"] == 1].copy()
    negatives = df.loc[df["target"] == 0].copy()

    normal_negatives = negatives.loc[~negatives["any_abnormal"]].copy()
    abnormal_negatives = negatives.loc[negatives["any_abnormal"]].copy()

    normal_eval_df = pd.concat([positives, normal_negatives], axis=0).copy()
    abnormal_eval_df = pd.concat([positives, abnormal_negatives], axis=0).copy()

    res_normal = evaluate_subset("pneumonia_vs_normal_negatives", normal_eval_df)
    res_abnormal = evaluate_subset("pneumonia_vs_abnormal_negatives", abnormal_eval_df)

    summary = {
        "predictions_csv": str(pred_path),
        "chexpert_csv": str(chex_path),
        "merge_keys": merge_keys,
        "abnormal_cols": ABNORMAL_COLS,
        "initial_rows_after_merge": int(initial_n),
        "rows_after_drop_all_nan_abnormal_cols": int(after_nan_drop_n),
        "counts": {
            "positives": int(len(positives)),
            "negatives_total": int(len(negatives)),
            "normal_negatives": int(len(normal_negatives)),
            "abnormal_negatives": int(len(abnormal_negatives)),
        },
        "results": {
            "pneumonia_vs_normal_negatives": res_normal,
            "pneumonia_vs_abnormal_negatives": res_abnormal,
        },
        "delta_abnormal_minus_normal": {
            "auroc": float(res_abnormal["auroc"] - res_normal["auroc"]),
            "auprc": float(res_abnormal["auprc"] - res_normal["auprc"]),
        },
    }

    print("\n=== MERGE INFO ===")
    print(f"merge_keys: {merge_keys}")
    print(f"rows after merge: {initial_n}")
    print(f"rows after dropping all-NaN abnormal rows: {after_nan_drop_n}")

    print("\n=== COUNTS ===")
    print(f"positives: {len(positives)}")
    print(f"normal negatives: {len(normal_negatives)}")
    print(f"abnormal negatives: {len(abnormal_negatives)}")

    print("\n=== PNEUMONIA vs NORMAL NEGATIVES ===")
    print(json.dumps(res_normal, indent=2))

    print("\n=== PNEUMONIA vs ABNORMAL NEGATIVES ===")
    print(json.dumps(res_abnormal, indent=2))

    print("\n=== PERFORMANCE DROP (abnormal - normal) ===")
    print(json.dumps(summary["delta_abnormal_minus_normal"], indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved JSON to: {out_path}")


if __name__ == "__main__":
    main()