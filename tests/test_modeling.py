"""Model training/evaluation on the sample data and edge cases."""

import numpy as np
import pandas as pd
import pytest

from data_agent.datasets import STORE
from data_agent.modeling import ModelingError, evaluate_classification, evaluate_regression, prepare_features


def test_classification_beats_the_baseline():
    result = evaluate_classification(STORE.load("customer-churn.csv"), "churned")
    assert result["accuracy"] > result["baseline_accuracy"]
    assert result["classes"] == ["no", "yes"] and result["n_test"] == 160
    assert 0 < result["f1_macro"] <= 1


def test_regression_explains_most_of_the_variance():
    result = evaluate_regression(STORE.load("house-prices.csv"), "price_k")
    assert result["r2_score"] > 0.8
    assert result["n_train"] + result["n_test"] == 588  # 12 rows with gaps dropped
    assert "area_sqft" in result["top_features"]


def test_results_are_reproducible():
    df = STORE.load("house-prices.csv")
    assert evaluate_regression(df, "price_k") == evaluate_regression(df, "price_k")


def test_text_features_are_one_hot_encoded():
    X, y = prepare_features(STORE.load("customer-churn.csv"), "churned")
    assert "contract_two-year" in X.columns and "contract" not in X.columns
    assert len(X) == len(y) == 800


@pytest.mark.parametrize(
    "df, target, message",
    [
        (pd.DataFrame({"a": range(20), "b": range(20)}), "zzz", "not found"),
        (pd.DataFrame({"a": [1.0] * 5, "b": [1] * 5}), "b", "not enough"),
        (pd.DataFrame({"b": range(20)}), "b", "No feature columns"),
    ],
)
def test_bad_inputs_raise_clear_errors(df, target, message):
    with pytest.raises(ModelingError, match=message):
        evaluate_classification(df, target)


def test_single_class_target():
    df = pd.DataFrame({"a": np.arange(30), "label": ["x"] * 30})
    with pytest.raises(ModelingError, match="single class"):
        evaluate_classification(df, "label")


def test_regression_on_text_target_is_refused():
    with pytest.raises(ModelingError, match="not numeric"):
        evaluate_regression(STORE.load("customer-churn.csv"), "churned")
