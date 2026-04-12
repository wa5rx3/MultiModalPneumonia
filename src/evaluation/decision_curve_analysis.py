from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "model"


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path).copy()

    required = {"target", "pred_prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["target"] = pd.to_numeric(df["target"], errors="raise").astype(int)
    df["pred_prob"] = pd.to_numeric(df["pred_prob"], errors="raise").astype(float)

    if not df["target"].isin([0, 1]).all():
        raise ValueError(f"{path} contains target values outside {{0,1}}")

    if ((df["pred_prob"] < 0.0) | (df["pred_prob"] > 1.0)).any():
        raise ValueError(f"{path} contains pred_prob values outside [0,1]")

    if len(df) == 0:
        raise ValueError(f"{path} is empty")

    return df


def validate_shared_targets(model_tables: dict[str, pd.DataFrame]) -> np.ndarray:
    y_true_ref: np.ndarray | None = None
    ref_name: str | None = None

    for name, df in model_tables.items():
        y_true = df["target"].to_numpy(dtype=int)

        if y_true_ref is None:
            y_true_ref = y_true
            ref_name = name
            continue

        if len(y_true) != len(y_true_ref):
            raise ValueError(
                f"Target length mismatch between '{ref_name}' ({len(y_true_ref)}) "
                f"and '{name}' ({len(y_true)})"
            )

        if not np.array_equal(y_true_ref, y_true):
            raise ValueError(
                f"Target mismatch between '{ref_name}' and '{name}'. "
                "Make sure all prediction files correspond to the same test set rows."
            )

    if y_true_ref is None:
        raise ValueError("No model tables loaded.")

    return y_true_ref


def compute_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    n = len(y_true)
    prevalence = float(np.mean(y_true))

    rows: list[dict[str, Any]] = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        odds = t / (1.0 - t)
        net_benefit = (tp / n) - (fp / n) * odds
        standardized_net_benefit = net_benefit / prevalence if prevalence > 0 else np.nan

        rows.append(
            {
                "threshold": float(t),
                "net_benefit": float(net_benefit),
                "standardized_net_benefit": float(standardized_net_benefit),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "predicted_positive": int(tp + fp),
                "predicted_negative": int(tn + fn),
            }
        )

    return pd.DataFrame(rows)


def compute_treat_all(y_true: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    prevalence = float(np.mean(y_true))
    rows: list[dict[str, Any]] = []

    for t in thresholds:
        odds = t / (1.0 - t)
        net_benefit = prevalence - (1.0 - prevalence) * odds
        standardized_net_benefit = net_benefit / prevalence if prevalence > 0 else np.nan

        rows.append(
            {
                "threshold": float(t),
                "net_benefit": float(net_benefit),
                "standardized_net_benefit": float(standardized_net_benefit),
            }
        )

    return pd.DataFrame(rows)


def compute_treat_none(thresholds: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "threshold": thresholds.astype(float),
            "net_benefit": np.zeros_like(thresholds, dtype=float),
            "standardized_net_benefit": np.zeros_like(thresholds, dtype=float),
        }
    )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    prevalence = float(np.mean(y_true))
    n = len(y_true)
    rows: list[dict[str, Any]] = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        tp = int(np.sum((preds == 1) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        sensitivity = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        ppv = safe_divide(tp, tp + fp)
        npv = safe_divide(tn, tn + fn)
        predicted_positive_rate = safe_divide(tp + fp, n)
        predicted_negative_rate = safe_divide(tn + fn, n)

        odds = t / (1.0 - t)
        net_benefit = (tp / n) - (fp / n) * odds
        standardized_net_benefit = net_benefit / prevalence if prevalence > 0 else np.nan

        rows.append(
            {
                "threshold": float(t),
                "prevalence": prevalence,
                "n": int(n),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": ppv,
                "npv": npv,
                "predicted_positive_rate": predicted_positive_rate,
                "predicted_negative_rate": predicted_negative_rate,
                "net_benefit": float(net_benefit),
                "standardized_net_benefit": float(standardized_net_benefit),
            }
        )

    return pd.DataFrame(rows)


def plot_decision_curve(
    curves: dict[str, pd.DataFrame],
    treat_all: pd.DataFrame,
    treat_none: pd.DataFrame,
    output_path: Path,
    standardized: bool = False,
) -> None:
    metric_col = "standardized_net_benefit" if standardized else "net_benefit"
    ylabel = "Standardized Net Benefit" if standardized else "Net Benefit"
    title = "Decision Curve Analysis (Standardized)" if standardized else "Decision Curve Analysis"

    plt.figure(figsize=(8, 6))

    for name, df in curves.items():
        plt.plot(df["threshold"], df[metric_col], label=name, linewidth=2)

    plt.plot(
        treat_all["threshold"],
        treat_all[metric_col],
        linestyle="--",
        linewidth=1.75,
        label="Treat All",
    )
    plt.plot(
        treat_none["threshold"],
        treat_none[metric_col],
        linestyle="--",
        linewidth=1.75,
        label="Treat None",
    )

    all_model_values = np.concatenate([df[metric_col].to_numpy(dtype=float) for df in curves.values()])
    finite_values = all_model_values[np.isfinite(all_model_values)]
    upper = max(0.01, float(np.nanmax(finite_values)) * 1.15) if finite_values.size > 0 else 0.05

    plt.ylim(bottom=-0.05, top=upper)
    plt.xlabel("Threshold probability")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_threshold_list(raw: str) -> list[float]:
    values: list[float] = []

    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue

        value = float(item)
        if not (0.0 < value < 1.0):
            raise ValueError(f"Threshold must be in (0,1), got {value}")
        values.append(value)

    if not values:
        raise ValueError("No valid thresholds provided.")

    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/evaluation/dca",
    )
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("NAME", "CSV"),
        help="Example: --model 'Image-only' path/to/test_predictions.csv",
    )
    parser.add_argument(
        "--n-thresholds",
        type=int,
        default=99,
        help="Number of equally spaced thresholds in [0.01, 0.99].",
    )
    parser.add_argument(
        "--threshold-metrics",
        type=str,
        default="0.2,0.5,0.8",
        help="Comma-separated thresholds for threshold metrics tables.",
    )
    args = parser.parse_args()

    if not args.model:
        raise ValueError("Provide at least one --model argument.")

    if args.n_thresholds < 2:
        raise ValueError("--n-thresholds must be at least 2")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_metrics_list = parse_threshold_list(args.threshold_metrics)
    dca_thresholds = np.linspace(0.01, 0.99, args.n_thresholds)

    model_tables: dict[str, pd.DataFrame] = {}
    model_paths: dict[str, str] = {}

    for name, csv_path in args.model:
        if name in model_tables:
            raise ValueError(f"Duplicate model name: {name}")
        path = Path(csv_path)
        model_tables[name] = load_predictions(path)
        model_paths[name] = str(path)

    y_true_ref = validate_shared_targets(model_tables)
    prevalence = float(np.mean(y_true_ref))

    curves: dict[str, pd.DataFrame] = {}
    threshold_metrics_outputs: dict[str, pd.DataFrame] = {}
    summary: dict[str, Any] = {
        "n_models": len(model_tables),
        "n_samples": int(len(y_true_ref)),
        "prevalence": prevalence,
        "dca_thresholds": {
            "min": float(np.min(dca_thresholds)),
            "max": float(np.max(dca_thresholds)),
            "n_thresholds": int(len(dca_thresholds)),
        },
        "threshold_metrics_requested": threshold_metrics_list,
        "models": {},
    }

    combined_curve_rows: list[pd.DataFrame] = []

    for name, df in model_tables.items():
        y_true = df["target"].to_numpy(dtype=int)
        y_prob = df["pred_prob"].to_numpy(dtype=float)

        curve_df = compute_net_benefit(y_true, y_prob, dca_thresholds)
        curve_df.insert(0, "model_name", name)
        curves[name] = curve_df
        combined_curve_rows.append(curve_df.copy())

        safe_name = sanitize_name(name)
        curve_df.to_csv(output_dir / f"{safe_name}_curve.csv", index=False)

        metrics_df = compute_threshold_metrics(y_true, y_prob, threshold_metrics_list)
        threshold_metrics_outputs[name] = metrics_df
        metrics_df.to_csv(output_dir / f"{safe_name}_threshold_metrics.csv", index=False)

        summary["models"][name] = {
            "predictions_file": model_paths[name],
            "n": int(len(y_true)),
            "positive_rate": float(np.mean(y_true)),
            "mean_pred_prob": float(np.mean(y_prob)),
            "threshold_metrics": metrics_df.to_dict(orient="records"),
        }

    treat_all = compute_treat_all(y_true_ref, dca_thresholds)
    treat_none = compute_treat_none(dca_thresholds)

    treat_all.to_csv(output_dir / "treat_all_curve.csv", index=False)
    treat_none.to_csv(output_dir / "treat_none_curve.csv", index=False)

    combined_curves_df = pd.concat(combined_curve_rows, ignore_index=True)
    combined_curves_df.to_csv(output_dir / "decision_curve_all_models.csv", index=False)

    plot_decision_curve(
        curves=curves,
        treat_all=treat_all,
        treat_none=treat_none,
        output_path=output_dir / "decision_curve.png",
        standardized=False,
    )

    plot_decision_curve(
        curves=curves,
        treat_all=treat_all,
        treat_none=treat_none,
        output_path=output_dir / "decision_curve_standardized.png",
        standardized=True,
    )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()