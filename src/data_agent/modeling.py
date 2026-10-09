"""Train-and-evaluate helpers (scikit-learn). Pure functions, no LLM involved."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


class ModelingError(Exception):
    pass


def prepare_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows with gaps and one-hot encode text columns, so any CSV can train."""
    if target_column not in df.columns:
        raise ModelingError(f"Target column '{target_column}' not found. Columns: {', '.join(df.columns)}.")
    clean = df.dropna()
    if len(clean) < 10:
        raise ModelingError(f"Only {len(clean)} complete rows - not enough to train a model.")
    features = clean.drop(columns=[target_column])
    if features.shape[1] == 0:
        raise ModelingError("No feature columns left besides the target.")
    X = pd.get_dummies(features, drop_first=True)
    return X, clean[target_column]


def _split(X, y, stratify=None):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify)


def evaluate_classification(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    X, y = prepare_features(df, target_column)
    if y.nunique() < 2:
        raise ModelingError(f"Target '{target_column}' has a single class - nothing to classify.")
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = _split(X, y, stratify)
    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE).fit(X_train, y_train)
    pred = model.predict(X_test)
    majority = float((y_test == y_train.mode()[0]).mean())
    return {
        "task": "classification",
        "target_column": target_column,
        "model": "RandomForestClassifier",
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "f1_macro": round(float(f1_score(y_test, pred, average="macro")), 4),
        "baseline_accuracy": round(majority, 4),  # always predicting the most common class
        "classes": sorted(map(str, y.unique())),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "top_features": _top_features(model, X.columns),
    }


def evaluate_regression(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    X, y = prepare_features(df, target_column)
    if not pd.api.types.is_numeric_dtype(y):
        raise ModelingError(f"Target '{target_column}' is not numeric - use classification instead.")
    X_train, X_test, y_train, y_test = _split(X, y)
    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE).fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "task": "regression",
        "target_column": target_column,
        "model": "RandomForestRegressor",
        "r2_score": round(float(r2_score(y_test, pred)), 4),
        "mean_squared_error": round(float(mean_squared_error(y_test, pred)), 4),
        "mean_absolute_error": round(float(mean_absolute_error(y_test, pred)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "top_features": _top_features(model, X.columns),
    }


def _top_features(model, columns, k: int = 3) -> list[str]:
    ranked = sorted(zip(model.feature_importances_, columns), reverse=True)[:k]
    return [name for _, name in ranked]
