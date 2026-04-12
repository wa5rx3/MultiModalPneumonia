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

from src.models.clinical_baseline_with_labs import (
    build_clinical_baseline_with_labs,
    prepare_feature_matrix,
)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/tables/cxr_clinical_labs_pneumonia_training_table_u_ignore_hadm_only_temporal.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/clinical_baseline_with_labs_u_ignore_hadm_only_temporal",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input).copy()

    if "temporal_split" not in df.columns:
        raise ValueError("Expected column 'temporal_split' not found.")

    train_df = df[df["temporal_split"] == "train"].copy()
    val_df = df[df["temporal_split"] == "validate"].copy()
    test_df = df[df["temporal_split"] == "test"].copy()

    bundle = build_clinical_baseline_with_labs()

    X_train = prepare_feature_matrix(train_df)
    y_train = train_df["target"].astype(int)

    X_val = prepare_feature_matrix(val_df)
    X_test = prepare_feature_matrix(test_df)

    bundle.pipeline.fit(X_train, y_train)

    val_prob = bundle.pipeline.predict_proba(X_val)[:, 1]
    test_prob = bundle.pipeline.predict_proba(X_test)[:, 1]

    val_metrics = evaluate_split(val_df, val_prob)
    test_metrics = evaluate_split(test_df, test_prob)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle.pipeline, output_dir / "model.joblib")

    val_out = val_df[["subject_id", "study_id", "target", "temporal_split"]].copy()
    val_out["pred_prob"] = val_prob
    val_out.to_csv(output_dir / "val_predictions.csv", index=False)

    test_out = test_df[["subject_id", "study_id", "target", "temporal_split"]].copy()
    test_out["pred_prob"] = test_prob
    test_out.to_csv(output_dir / "test_predictions.csv", index=False)

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