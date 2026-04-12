from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-csv",
        type=str,
        required=True,
        help="Path to predictions CSV with columns: target, pred_prob",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for confusion matrix / precision / recall",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs. Defaults to sibling folder next to predictions CSV.",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions_csv)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {pred_path}")

    if args.output_dir is None:
        output_dir = pred_path.parent / "prediction_behavior_check"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path).copy()

    required_cols = ["target", "pred_prob"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in predictions CSV: {missing}")

    df["target"] = pd.to_numeric(df["target"], errors="raise").astype(int)
    df["pred_prob"] = pd.to_numeric(df["pred_prob"], errors="raise").astype(float)

    if not ((df["pred_prob"] >= 0.0) & (df["pred_prob"] <= 1.0)).all():
        raise ValueError("pred_prob must be between 0 and 1.")

    y_true = df["target"].values
    y_prob = df["pred_prob"].values
    y_pred = (y_prob >= args.threshold).astype(int)

    unique_classes = np.unique(y_true)
    prevalence = float(y_true.mean())
    baseline_auprc = prevalence

    metrics = {
        "n": int(len(df)),
        "threshold": float(args.threshold),
        "prevalence": prevalence,
        "baseline_auprc_equals_prevalence": baseline_auprc,
        "mean_pred_prob": float(y_prob.mean()),
        "std_pred_prob": float(y_prob.std()),
        "min_pred_prob": float(y_prob.min()),
        "q01_pred_prob": float(np.quantile(y_prob, 0.01)),
        "q05_pred_prob": float(np.quantile(y_prob, 0.05)),
        "q25_pred_prob": float(np.quantile(y_prob, 0.25)),
        "median_pred_prob": float(np.quantile(y_prob, 0.50)),
        "q75_pred_prob": float(np.quantile(y_prob, 0.75)),
        "q95_pred_prob": float(np.quantile(y_prob, 0.95)),
        "q99_pred_prob": float(np.quantile(y_prob, 0.99)),
        "max_pred_prob": float(y_prob.max()),
        "predicted_positive_rate": float(y_pred.mean()),
        "predicted_negative_rate": float((1 - y_pred).mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(unique_classes) > 1 else None,
        "auprc": float(average_precision_score(y_true, y_prob)) if len(unique_classes) > 1 else None,
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]

    metrics["confusion_matrix"] = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    metrics["sanity_checks"] = {
        "all_positive_dummy_accuracy": prevalence,
        "all_positive_dummy_precision": prevalence,
        "all_positive_dummy_recall": 1.0,
        "all_positive_dummy_f1": (2 * prevalence) / (1 + prevalence) if prevalence > 0 else 0.0,
        "all_positive_dummy_auroc": 0.5,
        "all_positive_dummy_auprc": prevalence,
    }

    print("\n=== BASIC SUMMARY ===")
    print(f"n: {metrics['n']}")
    print(f"prevalence: {metrics['prevalence']:.6f}")
    print(f"baseline AUPRC (= prevalence): {metrics['baseline_auprc_equals_prevalence']:.6f}")

    print("\n=== PREDICTION DISTRIBUTION ===")
    print(f"mean_pred_prob: {metrics['mean_pred_prob']:.6f}")
    print(f"std_pred_prob: {metrics['std_pred_prob']:.6f}")
    print(f"min: {metrics['min_pred_prob']:.6f}")
    print(f"p01: {metrics['q01_pred_prob']:.6f}")
    print(f"p05: {metrics['q05_pred_prob']:.6f}")
    print(f"p25: {metrics['q25_pred_prob']:.6f}")
    print(f"p50: {metrics['median_pred_prob']:.6f}")
    print(f"p75: {metrics['q75_pred_prob']:.6f}")
    print(f"p95: {metrics['q95_pred_prob']:.6f}")
    print(f"p99: {metrics['q99_pred_prob']:.6f}")
    print(f"max: {metrics['max_pred_prob']:.6f}")

    print("\n=== MAIN METRICS ===")
    print(f"AUROC: {metrics['auroc']:.6f}" if metrics["auroc"] is not None else "AUROC: None")
    print(f"AUPRC: {metrics['auprc']:.6f}" if metrics["auprc"] is not None else "AUPRC: None")
    print(f"Accuracy @ {args.threshold:.2f}: {metrics['accuracy']:.6f}")
    print(f"Precision @ {args.threshold:.2f}: {metrics['precision']:.6f}")
    print(f"Recall @ {args.threshold:.2f}: {metrics['recall']:.6f}")
    print(f"F1 @ {args.threshold:.2f}: {metrics['f1']:.6f}")
    print(f"Predicted positive rate @ {args.threshold:.2f}: {metrics['predicted_positive_rate']:.6f}")

    print("\n=== CONFUSION MATRIX ===")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")

    print("\n=== DUMMY 'ALL POSITIVE' BASELINE ===")
    print(f"Accuracy: {metrics['sanity_checks']['all_positive_dummy_accuracy']:.6f}")
    print(f"Precision: {metrics['sanity_checks']['all_positive_dummy_precision']:.6f}")
    print(f"Recall: {metrics['sanity_checks']['all_positive_dummy_recall']:.6f}")
    print(f"F1: {metrics['sanity_checks']['all_positive_dummy_f1']:.6f}")
    print(f"AUROC: {metrics['sanity_checks']['all_positive_dummy_auroc']:.6f}")
    print(f"AUPRC: {metrics['sanity_checks']['all_positive_dummy_auprc']:.6f}")

    # Save metrics JSON-like CSV friendly summary
    pd.DataFrame(
        [
            {
                "n": metrics["n"],
                "threshold": metrics["threshold"],
                "prevalence": metrics["prevalence"],
                "baseline_auprc": metrics["baseline_auprc_equals_prevalence"],
                "mean_pred_prob": metrics["mean_pred_prob"],
                "std_pred_prob": metrics["std_pred_prob"],
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "predicted_positive_rate": metrics["predicted_positive_rate"],
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        ]
    ).to_csv(output_dir / "summary.csv", index=False)

    # Save full per-row predictions copy
    df.to_csv(output_dir / "predictions_copy.csv", index=False)

    # Histogram: positives vs negatives
    plt.figure(figsize=(8, 5))
    plt.hist(
        df.loc[df["target"] == 1, "pred_prob"],
        bins=30,
        alpha=0.5,
        label="Positive",
    )
    plt.hist(
        df.loc[df["target"] == 0, "pred_prob"],
        bins=30,
        alpha=0.5,
        label="Negative",
    )
    plt.axvline(args.threshold, linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Prediction histogram by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_histogram.png", dpi=180)
    plt.close()

    # Simple boxplot-style summary using pandas
    desc_df = df.groupby("target")["pred_prob"].describe().reset_index()
    desc_df.to_csv(output_dir / "prediction_distribution_by_class.csv", index=False)

    # High-confidence errors
    false_positives = df[(df["target"] == 0) & (df["pred_prob"] >= args.threshold)].copy()
    false_negatives = df[(df["target"] == 1) & (df["pred_prob"] < args.threshold)].copy()

    false_positives = false_positives.sort_values("pred_prob", ascending=False)
    false_negatives = false_negatives.sort_values("pred_prob", ascending=True)

    false_positives.head(100).to_csv(output_dir / "top_false_positives.csv", index=False)
    false_negatives.head(100).to_csv(output_dir / "top_false_negatives.csv", index=False)

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()