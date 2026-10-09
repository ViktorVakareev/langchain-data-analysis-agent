"""The LangChain tool wrappers: schemas the LLM sees, and errors returned as data."""

from data_agent.tools import (TOOLS, call_dataframe_method, evaluate_classification_dataset,
                              evaluate_regression_dataset, get_dataset_summaries, list_csv_files, preload_datasets)


def test_tool_names_descriptions_and_args():
    assert [t.name for t in TOOLS] == [
        "list_csv_files", "preload_datasets", "get_dataset_summaries", "call_dataframe_method",
        "evaluate_classification_dataset", "evaluate_regression_dataset"]
    for t in TOOLS:
        assert len(t.description) > 40, t.name
    assert list_csv_files.args == {}
    assert set(evaluate_regression_dataset.args) == {"file_name", "target_column"}


def test_list_csv_files_returns_none_when_empty(data_dir):
    for f in data_dir.glob("*.csv"):
        f.unlink()
    assert list_csv_files.invoke({}) is None


def test_happy_paths():
    assert list_csv_files.invoke({}) == ["customer-churn.csv", "house-prices.csv"]
    assert preload_datasets.invoke({"paths": ["house-prices.csv"]})["loaded"] == ["house-prices.csv"]
    assert get_dataset_summaries.invoke({"dataset_paths": ["house-prices.csv"]})[0]["rows"] == 600
    assert "price_k" in call_dataframe_method.invoke({"file_name": "house-prices.csv", "method": "head"})
    assert "accuracy" in evaluate_classification_dataset.invoke({"file_name": "customer-churn.csv", "target_column": "churned"})
    assert "r2_score" in evaluate_regression_dataset.invoke({"file_name": "house-prices.csv", "target_column": "price_k"})


def test_errors_are_returned_not_raised():
    assert get_dataset_summaries.invoke({"dataset_paths": ["nope.csv"]})[0]["error"].startswith("Dataset 'nope.csv' not found")
    assert call_dataframe_method.invoke({"file_name": "nope.csv", "method": "head"}).startswith("Error:")
    assert "not found" in evaluate_regression_dataset.invoke({"file_name": "house-prices.csv", "target_column": "colour"})["error"]
    assert "not numeric" in evaluate_regression_dataset.invoke({"file_name": "customer-churn.csv", "target_column": "churned"})["error"]
    assert "not found" in evaluate_classification_dataset.invoke({"file_name": "x.csv", "target_column": "y"})["error"]
