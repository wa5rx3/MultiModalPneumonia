"""
Plot ROC curves for all five canonical models on the temporal test set.

Reads test_predictions.csv from each model directory and saves:
    artifacts/evaluation/roc_curve_all_models.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "artifacts" / "evaluation" / "roc_curve_all_models.png"

MODELS: list[tuple[str, Path]] = [
    (
        "Clinical logistic",
        REPO_ROOT
        / "artifacts"
        / "models"
        / "clinical_baseline_u_ignore_temporal_strong_v2"
        / "test_predictions.csv",
    ),
    (
        "Clinical XGBoost",
        REPO_ROOT
        / "artifacts"
        / "models"
        / "clinical_xgb_u_ignore_temporal_strong_v2"
        / "test_predictions.csv",
    ),
    (
        "Image",
        REPO_ROOT
        / "artifacts"
        / "models"
        / "image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3"
        / "test_predictions.csv",
    ),
    (
        "Multimodal",
        REPO_ROOT
        / "artifacts"
        / "models"
        / "multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3"
        / "test_predictions.csv",
    ),
    (
        "Attention fusion",
        REPO_ROOT
        / "artifacts"
        / "models"
        / "multimodal_pneumonia_attn_fusion_u_ignore_temporal_v1"
        / "test_predictions.csv",
    ),
]

# Keep colors stable across plots / thesis figures
MODEL_COLORS: dict[str, str] = {
    "Clinical logistic": "#4c4c4c",
    "Clinical XGBoost": "#7a7a7a",
    "Image": "#1f77b4",
    "Multimodal": "#9467bd",
    "Attention fusion": "#d62728",
}

REQUIRED_COLS = {"subject_id", "study_id", "target", "pred_prob"}


def load_test_predictions(path: Path) -> pd.DataFrame:
    """Load and validate one model's test predictions."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    # If temporal_split exists, enforce test-only rows explicitly
    if "temporal_split" in df.columns:
        df = df.loc[df["temporal_split"] == "test"].copy()

    # Keep only columns needed for alignment + plotting
    df = df[["subject_id", "study_id", "target", "pred_prob"]].copy()

    # Normalize types
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="raise").astype("int64")
    df["study_id"] = pd.to_numeric(df["study_id"], errors="raise").astype("int64")
    df["target"] = pd.to_numeric(df["target"], errors="raise").astype("int64")
    df["pred_prob"] = pd.to_numeric(df["pred_prob"], errors="raise").astype("float64")

    # Basic sanity checks
    if df.empty:
        raise ValueError(f"{path} contains no rows after filtering.")

    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError(f"{path} contains non-binary targets: {sorted(df['target'].unique())}")

    if len(df["target"].unique()) < 2:
        raise ValueError(f"{path} contains only one class in the test set.")

    if ((df["pred_prob"] < 0) | (df["pred_prob"] > 1)).any():
        raise ValueError(f"{path} contains probabilities outside [0, 1].")

    # Prevent silent duplication / misalignment
    dup_mask = df.duplicated(subset=["subject_id", "study_id"], keep=False)
    if dup_mask.any():
        dup_rows = df.loc[dup_mask, ["subject_id", "study_id"]].drop_duplicates()
        raise ValueError(
            f"{path} contains duplicate (subject_id, study_id) keys. "
            f"Examples:\n{dup_rows.head()}"
        )

    df = df.sort_values(["subject_id", "study_id"]).reset_index(drop=True)
    return df


def align_models(model_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Check that all models cover the same test rows and labels."""
    names = list(model_frames.keys())
    ref_name = names[0]
    ref = model_frames[ref_name][["subject_id", "study_id", "target"]].copy()

    scores: dict[str, np.ndarray] = {}

    for name, df in model_frames.items():
        merged = ref.merge(
            df[["subject_id", "study_id", "target", "pred_prob"]],
            on=["subject_id", "study_id"],
            how="outer",
            indicator=True,
            suffixes=("_ref", "_cur"),
        )

        if not (merged["_merge"] == "both").all():
            bad = merged.loc[merged["_merge"] != "both", ["subject_id", "study_id", "_merge"]]
            raise ValueError(
                f"Model '{name}' does not align with reference model '{ref_name}'. "
                f"Mismatched keys examples:\n{bad.head()}"
            )

        if not (merged["target_ref"] == merged["target_cur"]).all():
            bad = merged.loc[
                merged["target_ref"] != merged["target_cur"],
                ["subject_id", "study_id", "target_ref", "target_cur"],
            ]
            raise ValueError(
                f"Model '{name}' has target mismatches versus reference '{ref_name}'. "
                f"Examples:\n{bad.head()}"
            )

        scores[name] = merged["pred_prob"].to_numpy(dtype=np.float64)

    return ref, scores


def plot_roc(y_true: np.ndarray, scores: dict[str, np.ndarray], out_path: Path) -> None:
    """Generate and save the ROC figure."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(6.8, 6.4), dpi=180)

    # Plot in intended order
    for name, _ in MODELS:
        y_score = scores[name]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        ax.plot(
            fpr,
            tpr,
            lw=2.2,
            color=MODEL_COLORS[name],
            label=f"{name} (AUROC = {auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", lw=1.2, label="Chance")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves on the Temporal Test Set")
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    model_frames: dict[str, pd.DataFrame] = {}

    for name, csv_path in MODELS:
        df = load_test_predictions(csv_path)
        model_frames[name] = df
        print(f"[OK] {name}: {len(df)} rows from {csv_path}")

    aligned_ref, scores = align_models(model_frames)
    y_true = aligned_ref["target"].to_numpy(dtype=np.int64)

    print(f"[OK] All models aligned on {len(aligned_ref)} test studies.")
    print(
        f"[INFO] Positives: {int(y_true.sum())} / {len(y_true)} "
        f"({y_true.mean():.3f} prevalence)"
    )

    plot_roc(y_true, scores, OUT_PATH)
    print(f"[DONE] Saved ROC figure to: {OUT_PATH}")


if __name__ == "__main__":
    main()