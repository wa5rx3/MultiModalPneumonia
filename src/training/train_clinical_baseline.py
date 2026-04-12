from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from src.models.clinical_baseline import build_clinical_baseline, prepare_feature_matrix


def evaluate_split(df: pd.DataFrame, y_prob, threshold: float = 0.5) -> dict:
    y_true = df["target"].astype(int).values
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "n": int(len(df)),
        "positive_rate": float(df["target"].mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "auprc": float(average_precision_score(y_true, y_prob)) if len(set(y_true)) > 1 else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def build_prediction_df(df: pd.DataFrame, y_prob) -> pd.DataFrame:
    id_cols = ["subject_id"]
    if "study_id" in df.columns:
        id_cols.append("study_id")
    if "dicom_id" in df.columns:
        id_cols.append("dicom_id")
    if "temporal_split" in df.columns:
        id_cols.append("temporal_split")

    out = df[id_cols].copy()
    out["target"] = df["target"].astype(int).values
    out["pred_prob"] = y_prob
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/manifests/cxr_clinical_pneumonia_training_table_u_ignore_temporal.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/clinical_baseline_u_ignore_temporal_strong_v2",
    )
    parser.add_argument(
        "--feature-groups",
        type=str,
        default="all",
        choices=["all", "vitals_only", "demographics_only", "acuity_only", "vitals_plus_acuity", "no_missing_flags"],
        dest="feature_groups",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input).copy()

    required_cols = ["subject_id", "target", "temporal_split"]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    train_df = df[df["temporal_split"] == "train"].copy()
    val_df = df[df["temporal_split"] == "validate"].copy()
    test_df = df[df["temporal_split"] == "test"].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError(
            "One or more temporal splits are empty. "
            f"train={len(train_df)}, validate={len(val_df)}, test={len(test_df)}"
        )

    bundle = build_clinical_baseline(feature_groups=args.feature_groups)

    X_train = prepare_feature_matrix(train_df, feature_groups=args.feature_groups)
    y_train = train_df["target"].astype(int)

    X_val = prepare_feature_matrix(val_df, feature_groups=args.feature_groups)
    X_test = prepare_feature_matrix(test_df, feature_groups=args.feature_groups)

    bundle.pipeline.fit(X_train, y_train)

    val_prob = bundle.pipeline.predict_proba(X_val)[:, 1]
    test_prob = bundle.pipeline.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_split(val_df, val_prob)
    test_metrics = evaluate_split(test_df, test_prob)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, output_dir / "model_bundle.joblib")
    joblib.dump(bundle.pipeline, output_dir / "model.joblib")

    val_out = build_prediction_df(val_df, val_prob)
    val_out.to_csv(output_dir / "val_predictions.csv", index=False)

    test_out = build_prediction_df(test_df, test_prob)
    test_out.to_csv(output_dir / "test_predictions.csv", index=False)

    config = {
        "input": args.input,
        "output_dir": str(output_dir),
        "model_name": "clinical_logistic_regression",
        "feature_groups": args.feature_groups,
        "train_rows": int(len(train_df)),
        "validate_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    metrics = {
        "input_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "validate_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()