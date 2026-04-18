"""Temperature scaling calibration for the image-only DenseNet-121 model.

Fits a single temperature scalar T on the validation set by minimising
negative log-likelihood, then applies it to the test set and reports ECE
with a patient-level bootstrap CI (B=2000, seed=42).

Usage:
    python scripts/temperature_scaling.py

Output:
    artifacts/evaluation/temperature_scaling_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid


sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.calibration_analysis import (
    bootstrap_metric_ci,
    compute_ece_mce,
    load_prediction_table,
)


MODEL_DIR = Path(
    "artifacts/models/"
    "image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3"
)
VAL_CSV = MODEL_DIR / "val_predictions.csv"
TEST_CSV = MODEL_DIR / "test_predictions.csv"
OUTPUT_JSON = Path("artifacts/evaluation/temperature_scaling_results.json")


IMAGE_ONLY_ORIGINAL_ECE = 0.0674
N_BINS = 10
N_BOOTSTRAP = 2000
SEED = 42


def prob_to_logit(p: np.ndarray) -> np.ndarray:
    """Convert probabilities to logits, clamping to avoid log(0)."""
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def nll_loss(T: float, logits: np.ndarray, targets: np.ndarray) -> float:
    """Negative log-likelihood under temperature-scaled sigmoid."""
    scaled = sigmoid(logits / T)
    scaled = np.clip(scaled, 1e-7, 1.0 - 1e-7)
    return -float(
        np.mean(targets * np.log(scaled) + (1.0 - targets) * np.log(1.0 - scaled))
    )


def main() -> None:

    print(f"Loading validation predictions from {VAL_CSV} …")
    df_val = load_prediction_table(VAL_CSV)
    y_val = df_val["target"].to_numpy(dtype=float)
    logits_val = prob_to_logit(df_val["pred_prob"].to_numpy(dtype=float))


    result = minimize_scalar(
        fun=lambda T: nll_loss(T, logits_val, y_val),
        bounds=(0.01, 20.0),
        method="bounded",
        options={"xatol": 1e-6},
    )
    T_opt = float(result.x)
    print(f"Optimal temperature T = {T_opt:.4f}  (val NLL before={nll_loss(1.0, logits_val, y_val):.4f}, after={result.fun:.4f})")


    print(f"Loading test predictions from {TEST_CSV} …")
    df_test = load_prediction_table(TEST_CSV)
    y_test = df_test["target"].to_numpy(dtype=float)
    logits_test = prob_to_logit(df_test["pred_prob"].to_numpy(dtype=float))
    scaled_probs = sigmoid(logits_test / T_opt).astype(float)


    ece_scaled_point, _, _ = compute_ece_mce(
        y_true=y_test.astype(int),
        y_prob=scaled_probs,
        n_bins=N_BINS,
    )
    print(f"Image-only ECE (original): {IMAGE_ONLY_ORIGINAL_ECE:.4f}")
    print(f"Image-only ECE (T-scaled):  {ece_scaled_point:.4f}")


    patient_ids: np.ndarray | None = None
    if "subject_id" in df_test.columns:
        patient_ids = df_test["subject_id"].to_numpy()

    print(f"Running patient-level bootstrap (B={N_BOOTSTRAP}, seed={SEED}) …")
    bootstrap_result = bootstrap_metric_ci(
        y_true=y_test.astype(int),
        y_prob=scaled_probs,
        metric_name="ece",
        n_bootstrap=N_BOOTSTRAP,
        seed=SEED,
        n_bins=N_BINS,
        patient_ids=patient_ids,
    )

    print(
        f"ECE (T-scaled) bootstrap: mean={bootstrap_result['mean']:.4f}, "
        f"95% CI [{bootstrap_result['ci_low']:.4f}, {bootstrap_result['ci_high']:.4f}]"
    )


    output: dict = {
        "method": "temperature_scaling",
        "model": "image_only_densenet121_u_ignore_temporal_stronger_lr_v3",
        "val_n": int(len(df_val)),
        "test_n": int(len(df_test)),
        "n_bins": N_BINS,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
        "optimal_temperature": round(T_opt, 6),
        "val_nll_before_scaling": round(nll_loss(1.0, logits_val, y_val), 6),
        "val_nll_after_scaling": round(float(result.fun), 6),
        "image_only_original_ece": IMAGE_ONLY_ORIGINAL_ECE,
        "image_only_scaled_ece_point": round(ece_scaled_point, 6),
        "image_only_scaled_ece_bootstrap_mean": round(bootstrap_result["mean"], 6),
        "image_only_scaled_ece_ci_low": round(bootstrap_result["ci_low"], 6),
        "image_only_scaled_ece_ci_high": round(bootstrap_result["ci_high"], 6),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {OUTPUT_JSON}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
