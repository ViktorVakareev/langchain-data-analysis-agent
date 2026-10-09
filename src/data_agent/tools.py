"""The six LangChain tools the agent can call.

Each tool is a thin ``@tool`` wrapper around the plain functions in ``datasets.py``
and ``modeling.py``. The docstring and type hints are what the LLM reads to decide
which tool to use and what to pass. Errors are returned as data, never raised.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool

from .datasets import ALLOWED_METHODS, STORE, DatasetError
from .modeling import ModelingError, evaluate_classification, evaluate_regression


@tool
def list_csv_files() -> Optional[list[str]]:
    """List the CSV dataset files available for analysis.

    Call this first when the user doesn't name a file.
    Returns a list of file names, or None if the data folder has no CSV files.
    """
    return STORE.list_files() or None


@tool
def preload_datasets(paths: list[str]) -> dict[str, list[str]]:
    """Load CSV files into the in-memory cache so later tools can use them by name.

    Args:
        paths: CSV file names, e.g. ["house-prices.csv"].

    Returns:
        {"loaded": [...], "already_cached": [...], "errors": [...]}
    """
    return STORE.preload(paths)


@tool
def get_dataset_summaries(dataset_paths: list[str]) -> list[dict[str, Any]]:
    """Describe the structure of one or more datasets without sending the data itself.

    Args:
        dataset_paths: CSV file names.

    Returns:
        One summary per file: {"file_name", "rows", "columns": [{"name", "dtype",
        "n_unique", "missing"}, ...]} or {"file_name", "error"}.
        Use dtype and n_unique to decide whether a column is a classification
        target (text or few distinct values) or a regression target (continuous numbers).
    """
    summaries = []
    for path in dataset_paths:
        try:
            summaries.append(STORE.summary(path))
        except DatasetError as exc:
            summaries.append({"file_name": path, "error": str(exc)})
    return summaries


@tool
def call_dataframe_method(file_name: str, method: str) -> str:
    """Run a read-only pandas inspection on a dataset and return the text output.

    Args:
        file_name: CSV file name, e.g. "customer-churn.csv".
        method: one of: head, tail, describe, info, shape, columns, dtypes,
            nunique, missing, corr.

    Returns:
        The formatted result, or an error message.
    """
    try:
        return STORE.run_method(file_name, method)
    except DatasetError as exc:
        return f"Error: {exc}"


@tool
def evaluate_classification_dataset(file_name: str, target_column: str) -> dict[str, Any]:
    """Train and evaluate a RandomForest classifier to predict a categorical column.

    Use when the target is text/boolean or has only a few distinct values
    (e.g. "yes"/"no", "churned", "species").

    Args:
        file_name: CSV file name.
        target_column: the column to predict.

    Returns:
        {"accuracy", "f1_macro", "baseline_accuracy", "classes", "n_train",
        "n_test", "top_features", ...} or {"error": "..."}.
    """
    try:
        return evaluate_classification(STORE.load(file_name), target_column)
    except (DatasetError, ModelingError) as exc:
        return {"error": str(exc)}


@tool
def evaluate_regression_dataset(file_name: str, target_column: str) -> dict[str, Any]:
    """Train and evaluate a RandomForest regressor to predict a continuous numeric column.

    Use when the target is a number with many distinct values (e.g. price, sales).

    Args:
        file_name: CSV file name.
        target_column: the column to predict.

    Returns:
        {"r2_score", "mean_squared_error", "mean_absolute_error", "n_train",
        "n_test", "top_features", ...} or {"error": "..."}.
    """
    try:
        return evaluate_regression(STORE.load(file_name), target_column)
    except (DatasetError, ModelingError) as exc:
        return {"error": str(exc)}


TOOLS = [
    list_csv_files,
    preload_datasets,
    get_dataset_summaries,
    call_dataframe_method,
    evaluate_classification_dataset,
    evaluate_regression_dataset,
]
TOOL_NAMES = [t.name for t in TOOLS]
METHODS = list(ALLOWED_METHODS)
