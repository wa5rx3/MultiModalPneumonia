from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRIAGE_NUMERIC_COLS: List[str] = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "acuity",
    "temperature_missing",
    "heartrate_missing",
    "resprate_missing",
    "o2sat_missing",
    "sbp_missing",
    "dbp_missing",
    "pain_missing",
    "acuity_missing",
    "is_pa",
    "is_ap",
]

TRIAGE_CATEGORICAL_COLS: List[str] = [
    "gender",
    "race",
    "arrival_transport",
]

LAB_NUMERIC_COLS: List[str] = [
    "albumin",
    "alkaline_phosphatase",
    "alt",
    "anion_gap",
    "ast",
    "base_excess",
    "bicarbonate",
    "bilirubin_total",
    "bun",
    "calcium",
    "chloride",
    "creatinine",
    "crp",
    "glucose",
    "hematocrit",
    "hemoglobin",
    "lactate",
    "pco2",
    "ph",
    "platelets",
    "po2",
    "potassium",
    "procalcitonin",
    "sodium",
    "total_protein",
    "wbc",
]

LAB_MISSING_FLAG_COLS: List[str] = [
    f"{col}_missing" for col in LAB_NUMERIC_COLS
]

NUMERIC_COLS: List[str] = TRIAGE_NUMERIC_COLS + LAB_NUMERIC_COLS + LAB_MISSING_FLAG_COLS
CATEGORICAL_COLS: List[str] = TRIAGE_CATEGORICAL_COLS


@dataclass
class ClinicalBaselineWithLabsBundle:
    pipeline: Pipeline
    feature_columns_numeric: List[str]
    feature_columns_categorical: List[str]


def build_clinical_baseline_with_labs() -> ClinicalBaselineWithLabsBundle:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return ClinicalBaselineWithLabsBundle(
        pipeline=pipeline,
        feature_columns_numeric=NUMERIC_COLS,
        feature_columns_categorical=CATEGORICAL_COLS,
    )


def prepare_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    needed = NUMERIC_COLS + CATEGORICAL_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X = df[needed].copy()

    # Ensure numeric columns are numeric
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Critical: left-merged rows may have null missing-flags. Make them explicit.
    for col in LAB_MISSING_FLAG_COLS:
        X[col] = X[col].fillna(True).astype(int)

    # Triage missing flags should also be explicit
    triage_flag_cols = [c for c in TRIAGE_NUMERIC_COLS if c.endswith("_missing")]
    for col in triage_flag_cols:
        if col in X.columns:
            X[col] = X[col].fillna(True).astype(int)

    return X