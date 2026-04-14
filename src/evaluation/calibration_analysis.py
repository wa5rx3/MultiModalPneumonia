from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def sanitize_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_prediction_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    df = pd.read_csv(path)

    required = {"target", "pred_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["target"] = pd.to_numeric(df["target"], errors="raise").astype(int)
    df["pred_prob"] = pd.to_numeric(df["pred_prob"], errors="raise").astype(float)

    if np.any(df["pred_prob"] < 0.0) or np.any(df["pred_prob"] > 1.0):
        raise ValueError(f"{path} contains pred_prob outside [0, 1].")

    return df


def compute_ece_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, pd.DataFrame]:
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Empty inputs.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=False)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        count = int(mask.sum())

        bin_left = float(bin_edges[bin_idx])
        bin_right = float(bin_edges[bin_idx + 1])
        bin_center = float((bin_left + bin_right) / 2.0)

        if count == 0:
            rows.append(
                {
                    "bin": bin_idx,
                    "bin_left": bin_left,
                    "bin_right": bin_right,
                    "bin_center": bin_center,
                    "count": 0,
                    "fraction": 0.0,
                    "avg_confidence": None,
                    "avg_accuracy": None,
                    "abs_gap": None,
                }
            )
            continue

        avg_confidence = float(y_prob[mask].mean())
        avg_accuracy = float(y_true[mask].mean())
        abs_gap = float(abs(avg_accuracy - avg_confidence))
        fraction = float(count / n)

        ece += fraction * abs_gap
        mce = max(mce, abs_gap)

        rows.append(
            {
                "bin": bin_idx,
                "bin_left": bin_left,
                "bin_right": bin_right,
                "bin_center": bin_center,
                "count": count,
                "fraction": fraction,
                "avg_confidence": avg_confidence,
                "avg_accuracy": avg_accuracy,
                "abs_gap": abs_gap,
            }
        )

    return float(ece), float(mce), pd.DataFrame(rows)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
    n_bins: int = 10,
    patient_ids: np.ndarray | None = None,
) -> dict[str, float]:
    """Bootstrap CI for ECE or Brier score.

    When *patient_ids* is provided the resampling unit is the patient (cluster),
    not the individual row.  All rows belonging to a resampled patient are
    included together, which is the correct approach when a patient may
    contribute more than one study.  This matches the patient-level bootstrap
    used for AUROC/AUPRC in bootstrap_eval.py.
    """
    rng = np.random.default_rng(seed)
    values: list[float] = []

    if patient_ids is not None:
        # --- patient-level resampling ---
        unique_patients = np.array(sorted(set(patient_ids.tolist())))
        # Build index arrays grouped by patient once (avoids repeated masking)
        patient_to_indices: dict[Any, np.ndarray] = {}
        for pid in unique_patients:
            patient_to_indices[pid] = np.where(patient_ids == pid)[0]

        for _ in range(n_bootstrap):
            sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
            idx = np.concatenate([patient_to_indices[pid] for pid in sampled])
            y_true_b = y_true[idx]
            y_prob_b = y_prob[idx]
            if len(np.unique(y_true_b)) < 2:
                continue  # skip degenerate replicate
            if metric_name == "brier":
                value = float(brier_score_loss(y_true_b, y_prob_b))
            elif metric_name == "ece":
                value, _, _ = compute_ece_mce(y_true_b, y_prob_b, n_bins=n_bins)
            else:
                raise ValueError(f"Unsupported metric_name: {metric_name}")
            values.append(value)
    else:
        # --- row-level resampling (legacy fallback) ---
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            y_true_b = y_true[idx]
            y_prob_b = y_prob[idx]
            if metric_name == "brier":
                value = float(brier_score_loss(y_true_b, y_prob_b))
            elif metric_name == "ece":
                value, _, _ = compute_ece_mce(y_true_b, y_prob_b, n_bins=n_bins)
            else:
                raise ValueError(f"Unsupported metric_name: {metric_name}")
            values.append(value)

    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


def calibration_metrics_from_predictions(
    df: pd.DataFrame,
    n_bins: int = 10,
    bootstrap: bool = False,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_true = df["target"].to_numpy(dtype=int)
    y_prob = df["pred_prob"].to_numpy(dtype=float)

    # Use patient-level bootstrap when subject_id is available so that the
    # resampling unit matches the AUROC/AUPRC bootstrap in bootstrap_eval.py.
    patient_ids: Optional[np.ndarray] = None
    if "subject_id" in df.columns:
        patient_ids = df["subject_id"].to_numpy()

    brier = float(brier_score_loss(y_true, y_prob))
    ece, mce, bin_df = compute_ece_mce(y_true=y_true, y_prob=y_prob, n_bins=n_bins)

    n_patients = int(df["subject_id"].nunique()) if "subject_id" in df.columns else None

    metrics: dict[str, Any] = {
        "n": int(len(df)),
        "n_patients": n_patients,
        "positive_rate": float(y_true.mean()),
        "mean_pred_prob": float(y_prob.mean()),
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "n_bins": int(n_bins),
        "bootstrap_level": "patient" if patient_ids is not None else "row",
    }

    if bootstrap:
        metrics["brier_score_bootstrap"] = bootstrap_metric_ci(
            y_true=y_true,
            y_prob=y_prob,
            metric_name="brier",
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
            n_bins=n_bins,
            patient_ids=patient_ids,
        )
        metrics["ece_bootstrap"] = bootstrap_metric_ci(
            y_true=y_true,
            y_prob=y_prob,
            metric_name="ece",
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
            n_bins=n_bins,
            patient_ids=patient_ids,
        )

    return metrics, bin_df


def plot_reliability_diagram(
    all_bin_tables: dict[str, pd.DataFrame],
    output_path: Path,
    title: str = "Reliability Diagram",
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax1.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    max_fraction = 0.0

    for model_name, bin_df in all_bin_tables.items():
        valid = bin_df.dropna(subset=["avg_confidence", "avg_accuracy"]).copy()
        if valid.empty:
            continue

        ax1.plot(
            valid["avg_confidence"].to_numpy(),
            valid["avg_accuracy"].to_numpy(),
            marker="o",
            label=model_name,
        )

        ax2.plot(
            valid["bin_center"].to_numpy(),
            valid["fraction"].to_numpy(),
            marker="o",
            label=model_name,
        )

        if not valid["fraction"].empty:
            max_fraction = max(max_fraction, float(valid["fraction"].max()))

    ax1.set_ylabel("Observed frequency")
    ax1.set_title(title)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Bin fraction")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, max(0.05, max_fraction * 1.15))
    ax2.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_single_model_reliability(
    model_name: str,
    bin_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    ax1.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

    valid = bin_df.dropna(subset=["avg_confidence", "avg_accuracy"]).copy()
    if not valid.empty:
        ax1.plot(
            valid["avg_confidence"].to_numpy(),
            valid["avg_accuracy"].to_numpy(),
            marker="o",
            label=model_name,
        )
        ax2.bar(
            valid["bin_center"].to_numpy(),
            valid["fraction"].to_numpy(),
            width=1.0 / max(len(bin_df), 1) * 0.9,
            align="center",
        )

    ax1.set_ylabel("Observed frequency")
    ax1.set_title(f"Reliability Diagram: {model_name}")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Bin fraction")
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def default_model_map() -> dict[str, str]:
    return {
        "Clinical Logistic": "artifacts/models/clinical_baseline_u_ignore_temporal_strong_v2/test_predictions.csv",
        "Clinical XGBoost": "artifacts/models/clinical_xgb_u_ignore_temporal_strong_v2/test_predictions.csv",
        "Image-only DenseNet121": "artifacts/models/image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/test_predictions.csv",
        "Multimodal": "artifacts/models/multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/test_predictions.csv",
    }


def build_model_map_from_args(args_models: list[list[str]] | None) -> dict[str, str]:
    if not args_models:
        return default_model_map()

    model_map: dict[str, str] = {}
    for item in args_models:
        if len(item) != 2:
            raise ValueError("Each --model argument must provide exactly 2 values: MODEL_NAME PREDICTIONS_CSV")
        model_name, csv_path = item
        model_map[model_name] = csv_path
    return model_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation/calibration_stronger_lr_v3",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Compute bootstrap confidence intervals for Brier score and ECE.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("MODEL_NAME", "PREDICTIONS_CSV"),
        help="Optional repeated override/addition: --model 'Name' path/to/test_predictions.csv",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_map = build_model_map_from_args(args.model)

    summary: dict[str, Any] = {
        "n_bins": int(args.n_bins),
        "bootstrap": bool(args.bootstrap),
        "n_bootstrap": int(args.n_bootstrap),
        "bootstrap_seed": int(args.bootstrap_seed),
        "models": {},
    }

    all_bin_tables: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for model_name, csv_path in model_map.items():
        pred_path = Path(csv_path)
        df = load_prediction_table(pred_path)

        metrics, bin_df = calibration_metrics_from_predictions(
            df=df,
            n_bins=args.n_bins,
            bootstrap=args.bootstrap,
            n_bootstrap=args.n_bootstrap,
            bootstrap_seed=args.bootstrap_seed,
        )

        summary["models"][model_name] = {
            "predictions_file": str(pred_path),
            **metrics,
        }

        summary_row = {
            "model_name": model_name,
            "predictions_file": str(pred_path),
            "n": metrics["n"],
            "positive_rate": metrics["positive_rate"],
            "mean_pred_prob": metrics["mean_pred_prob"],
            "brier_score": metrics["brier_score"],
            "ece": metrics["ece"],
            "mce": metrics["mce"],
        }

        if args.bootstrap:
            summary_row["brier_ci_low"] = metrics["brier_score_bootstrap"]["ci_low"]
            summary_row["brier_ci_high"] = metrics["brier_score_bootstrap"]["ci_high"]
            summary_row["ece_ci_low"] = metrics["ece_bootstrap"]["ci_low"]
            summary_row["ece_ci_high"] = metrics["ece_bootstrap"]["ci_high"]

        summary_rows.append(summary_row)

        safe_name = sanitize_name(model_name)
        bin_csv_path = output_dir / f"{safe_name}_bins.csv"
        bin_df.to_csv(bin_csv_path, index=False)

        plot_single_model_reliability(
            model_name=model_name,
            bin_df=bin_df,
            output_path=output_dir / f"{safe_name}_reliability.png",
        )

        all_bin_tables[model_name] = bin_df

    summary_path = output_dir / "calibration_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    summary_csv_path = output_dir / "calibration_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv_path, index=False)

    combined_plot_path = output_dir / "reliability_diagram_all_models.png"
    plot_reliability_diagram(
        all_bin_tables=all_bin_tables,
        output_path=combined_plot_path,
        title="Reliability Diagram (All Models)",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved calibration metrics to: {summary_path}")
    print(f"Saved calibration summary CSV to: {summary_csv_path}")
    print(f"Saved combined reliability diagram to: {combined_plot_path}")


if __name__ == "__main__":
    main()