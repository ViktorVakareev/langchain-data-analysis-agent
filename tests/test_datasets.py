"""DatasetStore: discovery, path safety, caching, summaries, allowed methods."""

import pandas as pd
import pytest

from data_agent.datasets import ALLOWED_METHODS, STORE, DatasetError, DatasetStore, infer_task_type


def test_lists_only_csv_files(data_dir):
    (data_dir / "notes.txt").write_text("not a dataset")
    assert STORE.list_files() == ["customer-churn.csv", "house-prices.csv"]


def test_empty_or_missing_folder(tmp_path):
    assert DatasetStore(tmp_path / "nope").list_files() == []


@pytest.mark.parametrize("name", ["house-prices.csv", "data/house-prices.csv", r"C:\x\house-prices.csv", " house-prices.csv "])
def test_resolve_uses_base_name_inside_data_dir(name, data_dir):
    assert STORE.resolve(name) == data_dir / "house-prices.csv"


@pytest.mark.parametrize(
    "name, message",
    [("../secrets.csv", "not found"), ("missing.csv", "Available: customer-churn.csv, house-prices.csv"),
     ("house-prices.txt", "not a CSV"), ("", "Empty"), ("..", "outside")],
)
def test_resolve_rejects_bad_names(name, message):
    with pytest.raises(DatasetError, match=message):
        STORE.resolve(name)


def test_cache_loads_each_file_once():
    first = STORE.load("house-prices.csv")
    assert STORE.load("house-prices.csv") is first
    assert STORE.preload(["house-prices.csv", "customer-churn.csv", "x.csv"]) == {
        "loaded": ["customer-churn.csv"], "already_cached": ["house-prices.csv"],
        "errors": ["Dataset 'x.csv' not found. Available: customer-churn.csv, house-prices.csv."]}


def test_unreadable_csv_is_reported(data_dir):
    (data_dir / "broken.csv").write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(DatasetError, match="Could not read"):
        STORE.load("broken.csv")


def test_summary_has_columns_types_uniques_and_gaps():
    summary = STORE.summary("house-prices.csv")
    assert summary["rows"] == 600
    cols = {c["name"]: c for c in summary["columns"]}
    assert cols["age_years"]["missing"] == 12
    assert cols["garage"]["n_unique"] == 2
    assert cols["price_k"]["dtype"] == "float64"


@pytest.mark.parametrize("method", list(ALLOWED_METHODS))
def test_every_allowed_method_returns_text(method):
    out = STORE.run_method("customer-churn.csv", method)
    assert isinstance(out, str) and out


def test_info_is_captured_not_printed(capsys):
    out = STORE.run_method("house-prices.csv", "info")
    assert "RangeIndex: 600 entries" in out and capsys.readouterr().out == ""


@pytest.mark.parametrize("method", ["to_csv", "drop", "__class__", "eval"])
def test_methods_outside_allow_list_are_refused(method):
    assert "is not supported" in STORE.run_method("house-prices.csv", method)


@pytest.mark.parametrize(
    "dtype, n_unique, rows, expected",
    [("str", 2, 800, "classification"), ("object", 5, 100, "classification"), ("bool", 2, 50, "classification"),
     ("int64", 3, 1000, "classification"), ("int64", 71, 800, "regression"), ("float64", 539, 600, "regression"),
     ("float64", 2, 600, "regression")],
)
def test_infer_task_type(dtype, n_unique, rows, expected):
    assert infer_task_type(dtype, n_unique, rows) == expected
