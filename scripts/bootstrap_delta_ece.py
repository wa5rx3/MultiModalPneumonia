"""Patient-level paired bootstrap for ΔECE (multimodal − image-only).

Generates a 95% CI and empirical P(ΔECE < 0) for the ECE difference,
making the calibration advantage claim statistically rigorous.

Sign convention (consistent with the AUROC delta in bootstrap_eval.py):
    ΔECE = ECE_multimodal − ECE_image
    ΔECE < 0  ⟹  multimodal is better calibrated (lower ECE)
    P(ΔECE < 0) is the one-tailed empirical probability

Usage:
    python scripts/bootstrap_delta_ece.py

Output:
    artifacts/evaluation/bootstrap_delta_ece.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.bootstrap_eval import assert_aligned_for_delta, load_predictions
from src.evaluation.calibration_analysis import compute_ece_mce


IMAGE_CSV = Path(
    "artifacts/models/"
    "image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/"
    "test_predictions.csv"
)
MULTI_CSV = Path(
    "artifacts/models/"
    "multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/"
    "test_predictions.csv"
)
OUTPUT_JSON = Path("artifacts/evaluation/bootstrap_delta_ece.json")

N_BOOTSTRAP = 2000
SEED = 42
N_BINS = 10


def main() -> None:

    print("Loading predictions …")
    df_image = load_predictions(str(IMAGE_CSV))
    df_multi = load_predictions(str(MULTI_CSV))

    print("Validating subject_id alignment …")
    df_image_aligned, df_multi_aligned, keys = assert_aligned_for_delta(df_image, df_multi)
    n = len(df_image_aligned)
    print(f"Aligned rows: {n}")

    y_true = df_image_aligned["target"].to_numpy(dtype=int)
    prob_image = df_image_aligned["prob"].to_numpy(dtype=float)
    prob_multi = df_multi_aligned["prob"].to_numpy(dtype=float)
    patient_ids = df_image_aligned["subject_id"].to_numpy(dtype=int)


    ece_image_point, _, _ = compute_ece_mce(y_true, prob_image, n_bins=N_BINS)
    ece_multi_point, _, _ = compute_ece_mce(y_true, prob_multi, n_bins=N_BINS)
    delta_point = ece_multi_point - ece_image_point
    print(f"Point ECE -- image: {ece_image_point:.4f}, multimodal: {ece_multi_point:.4f}, dECE: {delta_point:.4f}")


    rng = np.random.default_rng(SEED)
    unique_patients = np.array(sorted(set(patient_ids.tolist())))
    patient_to_indices: dict[int, np.ndarray] = {
        int(pid): np.where(patient_ids == pid)[0]
        for pid in unique_patients
    }

    delta_replicates: list[float] = []
    skipped = 0

    print(f"Running paired bootstrap (B={N_BOOTSTRAP}, seed={SEED}) …")
    for _ in range(N_BOOTSTRAP):
        sampled = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([patient_to_indices[int(pid)] for pid in sampled])

        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            skipped += 1
            continue

        ece_img_b, _, _ = compute_ece_mce(y_b, prob_image[idx], n_bins=N_BINS)
        ece_mul_b, _, _ = compute_ece_mce(y_b, prob_multi[idx], n_bins=N_BINS)
        delta_replicates.append(ece_mul_b - ece_img_b)

    arr = np.asarray(delta_replicates, dtype=float)
    delta_mean = float(arr.mean())
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    p_negative = float(np.mean(arr < 0.0))

    print(f"dECE bootstrap: mean={delta_mean:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}], P(dECE<0)={p_negative:.3f}")
    if skipped:
        print(f"Skipped {skipped} degenerate replicates (single class).")


    output: dict = {
        "description": "Paired patient-level bootstrap for delta ECE (multimodal - image-only). ΔECE < 0 means multimodal is better calibrated.",
        "model_a": "multimodal_concat",
        "model_b": "image_only",
        "n_test_rows": n,
        "n_unique_patients": int(len(unique_patients)),
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": SEED,
        "n_bins": N_BINS,
        "n_skipped_replicates": skipped,
        "ece_image_point": round(ece_image_point, 6),
        "ece_multimodal_point": round(ece_multi_point, 6),
        "delta_ece_point": round(delta_point, 6),
        "delta_ece_bootstrap_mean": round(delta_mean, 6),
        "delta_ece_ci_low": round(ci_low, 6),
        "delta_ece_ci_high": round(ci_high, 6),
        "p_delta_negative": round(p_negative, 4),
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results to {OUTPUT_JSON}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
