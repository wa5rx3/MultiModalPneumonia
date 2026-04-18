from __future__ import annotations

from typing import List

import pandas as pd
from xgboost import XGBClassifier


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

LAB_MISSING_FLAG_COLS: List[str] = [f"{col}_missing" for col in LAB_NUMERIC_COLS]

NUMERIC_COLS: List[str] = TRIAGE_NUMERIC_COLS + LAB_NUMERIC_COLS + LAB_MISSING_FLAG_COLS
CATEGORICAL_COLS: List[str] = TRIAGE_CATEGORICAL_COLS


def prepare_xgb_matrix(df: pd.DataFrame) -> pd.DataFrame:
    needed = NUMERIC_COLS + CATEGORICAL_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X = df[needed].copy()

    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")


    for col in LAB_MISSING_FLAG_COLS:
        X[col] = X[col].fillna(True).astype(int)

    triage_flag_cols = [c for c in TRIAGE_NUMERIC_COLS if c.endswith("_missing")]
    for col in triage_flag_cols:
        if col in X.columns:
            X[col] = X[col].fillna(True).astype(int)

    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")

    return X


def build_xgb_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
    )