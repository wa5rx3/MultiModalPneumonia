# Pipeline QC tests — skipped automatically if artifact files are not present.
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ARTIFACTS = Path("artifacts")
MANIFESTS = ARTIFACTS / "manifests"
TRAINING_TABLE = (
    MANIFESTS
    / "cxr_clinical_pneumonia_training_table_u_one_temporal.parquet"
)


def _load_training_table():
    """Load the primary training table; skip test if missing."""
    if not TRAINING_TABLE.exists():
        pytest.skip(f"Artifact not found: {TRAINING_TABLE}")
    return pd.read_parquet(TRAINING_TABLE)


class TestCohortIntegrity:
    """Verify cohort sizes match expected pipeline counts."""

    def test_training_table_row_count(self):
        df = _load_training_table()
        assert len(df) == 16963, f"Expected 16963 rows, got {len(df)}"

    def test_split_sizes(self):
        df = _load_training_table()
        counts = df["temporal_split"].value_counts()
        assert counts.get("train", 0) == 13342, f"train: {counts.get('train')}"
        assert counts.get("validate", 0) == 1677, f"validate: {counts.get('validate')}"
        assert counts.get("test", 0) == 1944, f"test: {counts.get('test')}"

    def test_positive_prevalence_test_set(self):
        df = _load_training_table()
        test_df = df[df["temporal_split"] == "test"]
        rate = test_df["target"].mean()
        assert 0.65 < rate < 0.75, f"Test prevalence {rate:.3f} outside expected range"


class TestNoPatientLeakage:
    """Verify no patient appears in more than one split."""

    def test_patient_split_exclusivity(self):
        df = _load_training_table()
        patient_splits = df.groupby("subject_id")["temporal_split"].nunique()
        leakers = patient_splits[patient_splits > 1]
        assert len(leakers) == 0, f"{len(leakers)} patients in multiple splits"


class TestTemporalConstraint:
    """Verify t0 within ED stay window for all rows."""

    def test_t0_within_stay(self):
        df = _load_training_table()
        required = {"t0", "intime", "outtime"}
        if not required.issubset(df.columns):
            pytest.skip(f"Columns {required - set(df.columns)} not in table")
        violations = df[(df["t0"] < df["intime"]) | (df["t0"] > df["outtime"])]
        assert len(violations) == 0, f"{len(violations)} rows violate t0 constraint"

    def test_no_missing_t0(self):
        df = _load_training_table()
        if "t0" not in df.columns:
            pytest.skip("t0 column not in training table")
        assert df["t0"].isna().sum() == 0, "Missing t0 values found"


class TestBootstrapAlignment:
    """Verify prediction files are aligned for paired delta."""

    IMAGE_PREDS = Path(
        "artifacts/models/"
        "image_pneumonia_finetune_densenet121_u_ignore_temporal_stronger_lr_v3/"
        "test_predictions.csv"
    )
    MULTI_PREDS = Path(
        "artifacts/models/"
        "multimodal_pneumonia_densenet121_triage_u_ignore_temporal_stronger_lr_v3/"
        "test_predictions.csv"
    )

    def test_prediction_alignment(self):
        if not self.IMAGE_PREDS.exists() or not self.MULTI_PREDS.exists():
            pytest.skip("Prediction files not found")
        df_a = pd.read_csv(self.IMAGE_PREDS)
        df_b = pd.read_csv(self.MULTI_PREDS)
        assert len(df_a) == len(df_b), (
            f"Length mismatch: {len(df_a)} vs {len(df_b)}"
        )
        if "subject_id" in df_a.columns and "subject_id" in df_b.columns:
            assert (
                df_a["subject_id"].values == df_b["subject_id"].values
            ).all(), "subject_id alignment failed"


class TestEvaluationArtifacts:
    """Verify key evaluation artifacts exist and are non-trivial."""

    @pytest.mark.parametrize(
        "path",
        [
            "artifacts/evaluation/final_publication_report.json",
            "artifacts/evaluation/feature_ablation_results.csv",
        ],
    )
    def test_artifact_exists(self, path):
        p = Path(path)
        if not p.exists():
            pytest.skip(f"Artifact not found: {path}")
        assert p.stat().st_size > 100, f"{path} is suspiciously small"

    def test_publication_report_valid_json(self):
        p = Path("artifacts/evaluation/final_publication_report.json")
        if not p.exists():
            pytest.skip("Publication report not found")
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "models" in data, "Missing 'models' key"
        assert "pairwise_comparisons" in data, "Missing 'pairwise_comparisons' key"
        assert "calibration" in data, "Missing 'calibration' key"

    def test_publication_report_model_count(self):
        p = Path("artifacts/evaluation/final_publication_report.json")
        if not p.exists():
            pytest.skip("Publication report not found")
        data = json.loads(p.read_text(encoding="utf-8"))
        assert len(data["models"]) >= 5, (
            f"Expected >= 5 models, got {len(data['models'])}"
        )
