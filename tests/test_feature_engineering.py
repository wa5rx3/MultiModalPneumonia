# Tests for clip_triage_vitals() and missingness flag logic. No MIMIC data needed.
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data.build_triage_features import clip_triage_vitals, TRIAGE_COLUMNS

# Must stay in sync with build_triage_features.py
CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (95.0, 105.8),
    "heartrate":   (30.0, 220.0),
    "resprate":    (5.0,  60.0),
    "o2sat":       (50.0, 100.0),
    "sbp":         (60.0, 250.0),
    "dbp":         (30.0, 150.0),
}

MISSINGNESS_COLS = ["temperature", "heartrate", "resprate", "o2sat",
                    "sbp", "dbp", "pain", "acuity"]


def _make_normal_row() -> pd.DataFrame:
    return pd.DataFrame([{
        "temperature": 98.6,
        "heartrate":   80.0,
        "resprate":    16.0,
        "o2sat":       98.0,
        "sbp":         120.0,
        "dbp":         80.0,
        "pain":        3.0,
        "acuity":      2.0,
    }])


def _add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MISSINGNESS_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna()
    return df


class TestVitalClipping(unittest.TestCase):

    def _clip_single(self, col: str, value: float) -> float:
        row = _make_normal_row()
        row[col] = value
        clipped = clip_triage_vitals(row)
        return float(clipped[col].iloc[0])

    def test_values_below_min_are_clipped(self):
        lo, _ = CLIP_BOUNDS["temperature"]
        result = self._clip_single("temperature", 50.0)
        self.assertAlmostEqual(result, lo)

    def test_values_above_max_are_clipped(self):
        _, hi = CLIP_BOUNDS["temperature"]
        result = self._clip_single("temperature", 110.0)
        self.assertAlmostEqual(result, hi)

    def test_values_within_bounds_unchanged(self):
        result = self._clip_single("temperature", 98.6)
        self.assertAlmostEqual(result, 98.6)

    def test_all_six_vitals_clipped_below(self):
        row = _make_normal_row()
        for col, (lo, _) in CLIP_BOUNDS.items():
            row[col] = lo - 999.0
        clipped = clip_triage_vitals(row)
        for col, (lo, _) in CLIP_BOUNDS.items():
            self.assertAlmostEqual(float(clipped[col].iloc[0]), lo,
                                   msg=f"{col} not clipped to lo={lo}")

    def test_all_six_vitals_clipped_above(self):
        row = _make_normal_row()
        for col, (_, hi) in CLIP_BOUNDS.items():
            row[col] = hi + 999.0
        clipped = clip_triage_vitals(row)
        for col, (_, hi) in CLIP_BOUNDS.items():
            self.assertAlmostEqual(float(clipped[col].iloc[0]), hi,
                                   msg=f"{col} not clipped to hi={hi}")

    def test_at_boundary_values_unchanged(self):
        """Exact boundary values should not be clipped."""
        for col, (lo, hi) in CLIP_BOUNDS.items():
            self.assertAlmostEqual(self._clip_single(col, lo), lo,
                                   msg=f"{col} lo boundary changed")
            self.assertAlmostEqual(self._clip_single(col, hi), hi,
                                   msg=f"{col} hi boundary changed")

    def test_heartrate_bounds(self):
        lo, hi = CLIP_BOUNDS["heartrate"]
        self.assertAlmostEqual(self._clip_single("heartrate", 5.0),  lo)
        self.assertAlmostEqual(self._clip_single("heartrate", 999.0), hi)

    def test_o2sat_bounds(self):
        lo, hi = CLIP_BOUNDS["o2sat"]
        self.assertAlmostEqual(self._clip_single("o2sat", 10.0),  lo)
        self.assertAlmostEqual(self._clip_single("o2sat", 110.0), hi)

    def test_returns_dataframe(self):
        row = _make_normal_row()
        result = clip_triage_vitals(row)
        self.assertIsInstance(result, pd.DataFrame)

    def test_nans_pass_through_unchanged(self):
        """NaN values should not be clipped (they remain NaN)."""
        row = _make_normal_row()
        row["temperature"] = np.nan
        clipped = clip_triage_vitals(row)
        self.assertTrue(np.isnan(float(clipped["temperature"].iloc[0])))


class TestMissingnessFlags(unittest.TestCase):

    def test_nan_produces_true_flag(self):
        row = _make_normal_row()
        row["temperature"] = np.nan
        flagged = _add_missingness_flags(row)
        self.assertTrue(bool(flagged["temperature_missing"].iloc[0]))

    def test_non_nan_produces_false_flag(self):
        row = _make_normal_row()
        flagged = _add_missingness_flags(row)
        self.assertFalse(bool(flagged["temperature_missing"].iloc[0]))

    def test_eight_flag_columns_created(self):
        row = _make_normal_row()
        flagged = _add_missingness_flags(row)
        flag_cols = [c for c in flagged.columns if c.endswith("_missing")]
        self.assertEqual(len(flag_cols), 8,
                         f"Expected 8 missingness flags, got {len(flag_cols)}: {flag_cols}")

    def test_flags_independent_per_column(self):
        """Only the column with NaN should get a True flag."""
        row = _make_normal_row()
        row["heartrate"] = np.nan
        flagged = _add_missingness_flags(row)
        self.assertTrue(bool(flagged["heartrate_missing"].iloc[0]))
        self.assertFalse(bool(flagged["temperature_missing"].iloc[0]))
        self.assertFalse(bool(flagged["o2sat_missing"].iloc[0]))

    def test_all_missing_all_flags_true(self):
        row = _make_normal_row()
        for col in MISSINGNESS_COLS:
            row[col] = np.nan
        flagged = _add_missingness_flags(row)
        for col in MISSINGNESS_COLS:
            self.assertTrue(bool(flagged[f"{col}_missing"].iloc[0]),
                            f"{col}_missing should be True")

    def test_no_missing_all_flags_false(self):
        row = _make_normal_row()
        flagged = _add_missingness_flags(row)
        for col in MISSINGNESS_COLS:
            self.assertFalse(bool(flagged[f"{col}_missing"].iloc[0]),
                             f"{col}_missing should be False")

    def test_pain_and_acuity_flagged(self):
        """Pain and acuity should also generate missingness flags."""
        row = _make_normal_row()
        row["pain"] = np.nan
        row["acuity"] = np.nan
        flagged = _add_missingness_flags(row)
        self.assertTrue(bool(flagged["pain_missing"].iloc[0]))
        self.assertTrue(bool(flagged["acuity_missing"].iloc[0]))

    def test_flag_dtype_is_bool(self):
        row = _make_normal_row()
        flagged = _add_missingness_flags(row)
        for col in MISSINGNESS_COLS:
            flag_col = f"{col}_missing"
            self.assertEqual(flagged[flag_col].dtype, bool,
                             f"{flag_col} dtype should be bool")


class TestTriageColumnsConstant(unittest.TestCase):

    def test_triage_columns_is_list(self):
        self.assertIsInstance(TRIAGE_COLUMNS, list)

    def test_triage_columns_contains_vitals(self):
        expected = {"temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"}
        self.assertTrue(expected.issubset(set(TRIAGE_COLUMNS)),
                        f"Missing vitals in TRIAGE_COLUMNS: {expected - set(TRIAGE_COLUMNS)}")


if __name__ == "__main__":
    unittest.main()
