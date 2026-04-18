from __future__ import annotations

from typing import List

import pandas as pd
from xgboost import XGBClassifier

from src.models.clinical_baseline import FEATURE_GROUP_COLUMNS, MISSING_INDICATOR_COLS


NUMERIC_COLS: List[str] = FEATURE_GROUP_COLUMNS["all"]["numeric"]
CATEGORICAL_COLS: List[str] = FEATURE_GROUP_COLUMNS["all"]["categorical"]


def prepare_xgb_matrix(df: pd.DataFrame, feature_groups: str = "all") -> pd.DataFrame:
    if feature_groups not in FEATURE_GROUP_COLUMNS:
        raise ValueError(
            f"Unknown feature_groups '{feature_groups}'. "
            f"Choose from: {list(FEATURE_GROUP_COLUMNS)}"
        )
    group = FEATURE_GROUP_COLUMNS[feature_groups]
    numeric_cols = group["numeric"]
    categorical_cols = group["categorical"]

    needed = numeric_cols + categorical_cols
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X = df[needed].copy()

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in MISSING_INDICATOR_COLS:
        if col in numeric_cols:
            X[col] = X[col].fillna(1).astype(int)

    for col in ["is_pa", "is_ap"]:
        if col in numeric_cols:
            X[col] = X[col].fillna(0).astype(int)

    for col in categorical_cols:
        X[col] = X[col].astype("string").fillna("UNKNOWN").str.strip()
        X[col] = X[col].replace({"": "UNKNOWN"})
        X[col] = X[col].astype("category")

    return X


def build_xgb_model(scale_pos_weight: float, early_stopping_rounds: int = 40) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        n_jobs=-1,
        early_stopping_rounds=early_stopping_rounds,
    )
