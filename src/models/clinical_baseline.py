from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_COLS: List[str] = [
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

CATEGORICAL_COLS: List[str] = [
    "gender",
    "race",
    "arrival_transport",
]

MISSING_INDICATOR_COLS: List[str] = [
    "temperature_missing",
    "heartrate_missing",
    "resprate_missing",
    "o2sat_missing",
    "sbp_missing",
    "dbp_missing",
    "pain_missing",
    "acuity_missing",
]

VITAL_COLS: List[str] = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
]

FEATURE_GROUP_COLUMNS: Dict[str, Dict[str, List[str]]] = {
    "all": {
        "numeric": NUMERIC_COLS,
        "categorical": CATEGORICAL_COLS,
    },
    "vitals_only": {
        "numeric": VITAL_COLS + ["is_pa", "is_ap"],
        "categorical": [],
    },
    "demographics_only": {
        "numeric": ["is_pa", "is_ap"],
        "categorical": CATEGORICAL_COLS,
    },
    "acuity_only": {
        "numeric": ["acuity", "is_pa", "is_ap"],
        "categorical": [],
    },
    "vitals_plus_acuity": {
        "numeric": VITAL_COLS + ["acuity", "is_pa", "is_ap"],
        "categorical": [],
    },
    "no_missing_flags": {
        "numeric": VITAL_COLS + ["acuity", "is_pa", "is_ap"],
        "categorical": CATEGORICAL_COLS,
    },
}


@dataclass
class ClinicalBaselineBundle:
    pipeline: Pipeline
    feature_columns_numeric: List[str]
    feature_columns_categorical: List[str]


def build_clinical_baseline(feature_groups: str = "all") -> ClinicalBaselineBundle:
    if feature_groups not in FEATURE_GROUP_COLUMNS:
        raise ValueError(
            f"Unknown feature_groups '{feature_groups}'. "
            f"Choose from: {list(FEATURE_GROUP_COLUMNS)}"
        )
    group = FEATURE_GROUP_COLUMNS[feature_groups]
    numeric_cols = group["numeric"]
    categorical_cols = group["categorical"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [("num", numeric_transformer, numeric_cols)]

    if categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    model = LogisticRegression(
        solver="saga",
        max_iter=10_000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return ClinicalBaselineBundle(
        pipeline=pipeline,
        feature_columns_numeric=numeric_cols,
        feature_columns_categorical=categorical_cols,
    )


def prepare_feature_matrix(df: pd.DataFrame, feature_groups: str = "all") -> pd.DataFrame:
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

    return X
