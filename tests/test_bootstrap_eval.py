# Tests for bootstrap evaluation logic using synthetic DataFrames. No MIMIC data needed.
from __future__ import annotations

import io
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.evaluation.bootstrap_eval import (
    assert_aligned_for_delta,
    bootstrap_delta,
    bootstrap_patient_level,
    compute_metrics,
    summarize_delta,
)


def _make_predictions(
    n_patients: int = 5,
    rows_per_patient: int = 4,
    seed: int = 0,
    perfect: bool = False,
    random_model: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    subject_ids = np.repeat(np.arange(1, n_patients + 1), rows_per_patient)
    targets = np.array([i % 2 for i in range(n_patients * rows_per_patient)])

    if perfect:
        probs = targets.astype(float)
    elif random_model:
        probs = rng.uniform(0, 1, size=len(subject_ids))
    else:
        probs = rng.uniform(0, 1, size=len(subject_ids))

    return pd.DataFrame({
        "subject_id": subject_ids,
        "target": targets,
        "prob": probs,
    })


def _make_df_with_col(col_name: str, values=None) -> pd.DataFrame:
    if values is None:
        values = [0.1, 0.9, 0.2, 0.8]
    return pd.DataFrame({
        "subject_id": [1, 1, 2, 2],
        "target":     [0, 1, 0, 1],
        col_name:     values,
    })


class TestComputeMetrics(unittest.TestCase):

    def test_perfect_classifier(self):
        df = _make_predictions(perfect=True)
        metrics = compute_metrics(df)
        self.assertAlmostEqual(metrics["auroc"], 1.0, places=6)
        self.assertAlmostEqual(metrics["auprc"], 1.0, places=6)

    def test_random_classifier_auroc_near_half(self):
        rng = np.random.default_rng(99)
        n = 2000
        targets = (np.arange(n) % 2).astype(int)
        probs = rng.uniform(0, 1, size=n)
        subject_ids = np.arange(n)
        df = pd.DataFrame({"subject_id": subject_ids, "target": targets, "prob": probs})
        metrics = compute_metrics(df)
        self.assertAlmostEqual(metrics["auroc"], 0.5, delta=0.05)

    def test_raises_on_single_class(self):
        df = pd.DataFrame({
            "subject_id": [1, 2, 3],
            "target":     [1, 1, 1],
            "prob":       [0.8, 0.9, 0.7],
        })
        with self.assertRaises(ValueError):
            compute_metrics(df)

    def test_auroc_and_auprc_keys_present(self):
        df = _make_predictions()
        metrics = compute_metrics(df)
        self.assertIn("auroc", metrics)
        self.assertIn("auprc", metrics)

    def test_auroc_in_valid_range(self):
        df = _make_predictions(seed=7)
        metrics = compute_metrics(df)
        self.assertGreaterEqual(metrics["auroc"], 0.0)
        self.assertLessEqual(metrics["auroc"], 1.0)

    def test_auprc_in_valid_range(self):
        df = _make_predictions(seed=7)
        metrics = compute_metrics(df)
        self.assertGreaterEqual(metrics["auprc"], 0.0)
        self.assertLessEqual(metrics["auprc"], 1.0)


class TestLoadPredictions(unittest.TestCase):

    def _write_csv(self, df: pd.DataFrame) -> str:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        )
        df.to_csv(tmp.name, index=False)
        tmp.close()
        return tmp.name

    def test_pred_prob_column_renamed_to_prob(self):
        from src.evaluation.bootstrap_eval import load_predictions
        df = _make_df_with_col("pred_prob")
        path = self._write_csv(df)
        loaded = load_predictions(path)
        self.assertIn("prob", loaded.columns)

    def test_prediction_column_renamed_to_prob(self):
        from src.evaluation.bootstrap_eval import load_predictions
        df = _make_df_with_col("prediction")
        path = self._write_csv(df)
        loaded = load_predictions(path)
        self.assertIn("prob", loaded.columns)

    def test_logit_column_converted_to_prob(self):
        from src.evaluation.bootstrap_eval import load_predictions
        df = _make_df_with_col("logit", values=[0.0, 2.0, -2.0, 1.0])
        path = self._write_csv(df)
        loaded = load_predictions(path)
        self.assertIn("prob", loaded.columns)
        self.assertAlmostEqual(float(loaded["prob"].iloc[0]), 0.5, places=5)  # sigmoid(0) = 0.5

    def test_missing_prob_column_raises(self):
        from src.evaluation.bootstrap_eval import load_predictions
        df = pd.DataFrame({
            "subject_id": [1, 2],
            "target":     [0, 1],
            "score_xyz":  [0.3, 0.7],
        })
        path = self._write_csv(df)
        with self.assertRaises(ValueError):
            load_predictions(path)

    def test_prob_column_kept_as_is(self):
        from src.evaluation.bootstrap_eval import load_predictions
        df = _make_df_with_col("prob")
        path = self._write_csv(df)
        loaded = load_predictions(path)
        self.assertIn("prob", loaded.columns)

    def test_subject_id_cast_to_int(self):
        from src.evaluation.bootstrap_eval import load_predictions
        df = _make_df_with_col("prob")
        path = self._write_csv(df)
        loaded = load_predictions(path)
        self.assertEqual(loaded["subject_id"].dtype, np.dtype("int64"))


class TestBootstrapPatientLevel(unittest.TestCase):

    def setUp(self):
        self.df = _make_predictions(n_patients=10, rows_per_patient=4, seed=42)
        self.B = 50  # small B for test speed

    def test_output_columns(self):
        boot_df, _ = bootstrap_patient_level(self.df, n_bootstrap=self.B, seed=0)
        self.assertIn("auroc", boot_df.columns)
        self.assertIn("auprc", boot_df.columns)

    def test_output_length_approximately_B(self):
        boot_df, skipped = bootstrap_patient_level(self.df, n_bootstrap=self.B, seed=0)
        self.assertEqual(len(boot_df) + skipped, self.B)

    def test_auroc_in_valid_range(self):
        boot_df, _ = bootstrap_patient_level(self.df, n_bootstrap=self.B, seed=0)
        self.assertTrue((boot_df["auroc"] >= 0.0).all())
        self.assertTrue((boot_df["auroc"] <= 1.0).all())

    def test_auprc_in_valid_range(self):
        boot_df, _ = bootstrap_patient_level(self.df, n_bootstrap=self.B, seed=0)
        self.assertTrue((boot_df["auprc"] >= 0.0).all())
        self.assertTrue((boot_df["auprc"] <= 1.0).all())

    def test_seed_reproducibility(self):
        boot1, _ = bootstrap_patient_level(self.df, n_bootstrap=self.B, seed=99)
        boot2, _ = bootstrap_patient_level(self.df, n_bootstrap=self.B, seed=99)
        pd.testing.assert_frame_equal(boot1, boot2)

    def test_different_seeds_different_results(self):
        boot1, _ = bootstrap_patient_level(self.df, n_bootstrap=self.B, seed=1)
        boot2, _ = bootstrap_patient_level(self.df, n_bootstrap=self.B, seed=2)
        self.assertNotAlmostEqual(
            boot1["auroc"].mean(), boot2["auroc"].mean(), places=8
        )


class TestBootstrapDelta(unittest.TestCase):

    def setUp(self):
        self.B = 50

    def _make_aligned_pair(self, seed_a=0, seed_b=1, perfect_a=False):
        n = 20
        subject_ids = np.arange(1, n + 1)
        targets = np.array([i % 2 for i in range(n)])
        rng_a = np.random.default_rng(seed_a)
        rng_b = np.random.default_rng(seed_b)
        probs_a = targets.astype(float) if perfect_a else rng_a.uniform(0, 1, n)
        probs_b = rng_b.uniform(0, 1, n)
        df_a = pd.DataFrame({"subject_id": subject_ids, "target": targets, "prob": probs_a})
        df_b = pd.DataFrame({"subject_id": subject_ids, "target": targets, "prob": probs_b})
        return df_a, df_b

    def test_delta_is_difference_of_metrics(self):
        df_a, df_b = self._make_aligned_pair()
        m_a = compute_metrics(df_a)
        m_b = compute_metrics(df_b)
        delta_df, _ = bootstrap_delta(df_a, df_b, n_bootstrap=self.B, seed=0)
        expected_sign = m_a["auroc"] - m_b["auroc"]
        observed_mean = delta_df["delta_auroc"].mean()
        self.assertEqual(np.sign(expected_sign), np.sign(observed_mean))

    def test_output_columns_delta(self):
        df_a, df_b = self._make_aligned_pair()
        delta_df, _ = bootstrap_delta(df_a, df_b, n_bootstrap=self.B, seed=0)
        self.assertIn("delta_auroc", delta_df.columns)
        self.assertIn("delta_auprc", delta_df.columns)

    def test_p_positive_perfect_vs_random(self):
        df_a, df_b = self._make_aligned_pair(perfect_a=True)
        delta_df, _ = bootstrap_delta(df_a, df_b, n_bootstrap=self.B, seed=0)
        summary = summarize_delta(delta_df)
        self.assertGreater(summary["delta_auroc"]["p_positive"], 0.8)

    def test_p_positive_random_vs_perfect(self):
        df_a, df_b = self._make_aligned_pair(perfect_a=True)
        delta_df, _ = bootstrap_delta(df_b, df_a, n_bootstrap=self.B, seed=0)
        summary = summarize_delta(delta_df)
        self.assertLess(summary["delta_auroc"]["p_positive"], 0.2)

    def test_skipped_count_non_negative(self):
        df_a, df_b = self._make_aligned_pair()
        _, skipped = bootstrap_delta(df_a, df_b, n_bootstrap=self.B, seed=0)
        self.assertGreaterEqual(skipped, 0)


class TestAssertAligned(unittest.TestCase):

    def _aligned_pair(self):
        ids = [1, 2, 3, 4, 5, 6]
        targets = [0, 1, 0, 1, 0, 1]
        df_a = pd.DataFrame({"subject_id": ids, "target": targets, "prob": [0.1]*6})
        df_b = pd.DataFrame({"subject_id": ids, "target": targets, "prob": [0.9]*6})
        return df_a, df_b

    def test_aligned_pair_returns_without_error(self):
        df_a, df_b = self._aligned_pair()
        result = assert_aligned_for_delta(df_a, df_b)
        self.assertEqual(len(result), 3)  # returns (aligned_a, aligned_b, keys)

    def test_raises_on_mismatched_subjects(self):
        df_a = pd.DataFrame({
            "subject_id": [1, 2, 3],
            "target": [0, 1, 0],
            "prob": [0.2, 0.8, 0.3],
        })
        df_b = pd.DataFrame({
            "subject_id": [1, 2, 4],  # subject 3 → 4 (mismatch)
            "target": [0, 1, 0],
            "prob": [0.3, 0.7, 0.4],
        })
        with self.assertRaises(ValueError):
            assert_aligned_for_delta(df_a, df_b)

    def test_raises_on_target_mismatch(self):
        """Same subjects but different targets → ValueError."""
        df_a = pd.DataFrame({
            "subject_id": [1, 2],
            "target": [0, 1],
            "prob": [0.3, 0.7],
        })
        df_b = pd.DataFrame({
            "subject_id": [1, 2],
            "target": [1, 0],  # swapped
            "prob": [0.3, 0.7],
        })
        with self.assertRaises(ValueError):
            assert_aligned_for_delta(df_a, df_b)

    def test_aligned_output_has_prob_column(self):
        df_a, df_b = self._aligned_pair()
        out_a, out_b, keys = assert_aligned_for_delta(df_a, df_b)
        self.assertIn("prob", out_a.columns)
        self.assertIn("prob", out_b.columns)

    def test_alignment_keys_include_subject_id(self):
        df_a, df_b = self._aligned_pair()
        _, _, keys = assert_aligned_for_delta(df_a, df_b)
        self.assertIn("subject_id", keys)


if __name__ == "__main__":
    unittest.main()
