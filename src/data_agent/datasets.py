"""Dataset discovery, caching and inspection - plain Python, no LLM involved.

The agent never sees whole datasets. It refers to them by **file name**, and the
``DatasetStore`` keeps the loaded DataFrames in memory between tool calls. That
saves tokens and context window, and each CSV is read only once.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

# Read-only inspection methods the agent may run. An allow-list instead of
# getattr(df, method), so the LLM can't call arbitrary DataFrame methods.
ALLOWED_METHODS: dict[str, Callable[[pd.DataFrame], Any]] = {
    "head": lambda df: df.head(),
    "tail": lambda df: df.tail(),
    "describe": lambda df: df.describe(include="all"),
    "info": lambda df: _info_text(df),
    "shape": lambda df: f"{df.shape[0]} rows x {df.shape[1]} columns",
    "columns": lambda df: list(df.columns),
    "dtypes": lambda df: df.dtypes.astype(str),
    "nunique": lambda df: df.nunique(),
    "missing": lambda df: df.isna().sum(),
    "corr": lambda df: df.select_dtypes("number").corr().round(3),
}


def _info_text(df: pd.DataFrame) -> str:
    # df.info() prints instead of returning - capture it
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


class DatasetError(Exception):
    """Raised for unknown, unreadable or out-of-folder datasets."""


class DatasetStore:
    """Finds CSV files in ``data_dir`` and caches them as DataFrames."""

    def __init__(self, data_dir: Optional[os.PathLike | str] = None):
        self.data_dir = Path(data_dir or os.getenv("DATA_DIR", "data")).resolve()
        self.cache: dict[str, pd.DataFrame] = {}

    # ---- discovery -----------------------------------------------------------
    def list_files(self) -> list[str]:
        if not self.data_dir.is_dir():
            return []
        return sorted(p.name for p in self.data_dir.glob("*.csv"))

    def resolve(self, name: str) -> Path:
        """Map a file name to a CSV inside ``data_dir``.

        Only the base name is used, so paths like "../secrets.csv" or
        "C:/other/file.csv" can never reach files outside the data folder.
        """
        if not isinstance(name, str) or not name.strip():
            raise DatasetError("Empty dataset name.")
        candidate = (self.data_dir / Path(name.strip().replace("\\", "/")).name).resolve()
        if candidate.parent != self.data_dir:
            raise DatasetError(f"'{name}' is outside the data folder.")
        if candidate.suffix.lower() != ".csv":
            raise DatasetError(f"'{name}' is not a CSV file.")
        if not candidate.is_file():
            available = ", ".join(self.list_files()) or "none"
            raise DatasetError(f"Dataset '{name}' not found. Available: {available}.")
        return candidate

    # ---- caching -------------------------------------------------------------
    def load(self, name: str) -> pd.DataFrame:
        key = self.resolve(name).name
        if key not in self.cache:
            try:
                self.cache[key] = pd.read_csv(self.data_dir / key)
            except Exception as exc:  # malformed file, encoding, ...
                raise DatasetError(f"Could not read '{key}': {exc}") from exc
        return self.cache[key]

    def preload(self, names: list[str]) -> dict[str, list[str]]:
        report: dict[str, list[str]] = {"loaded": [], "already_cached": [], "errors": []}
        for name in names:
            try:
                key = self.resolve(name).name
            except DatasetError as exc:
                report["errors"].append(str(exc))
                continue
            if key in self.cache:
                report["already_cached"].append(key)
            else:
                self.load(key)
                report["loaded"].append(key)
        return report

    def clear(self) -> None:
        self.cache.clear()

    # ---- inspection ----------------------------------------------------------
    def summary(self, name: str) -> dict[str, Any]:
        """Metadata the agent needs to plan: columns, types, unique counts, gaps."""
        df = self.load(name)
        return {
            "file_name": self.resolve(name).name,
            "rows": int(df.shape[0]),
            "columns": [
                {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "n_unique": int(df[col].nunique()),
                    "missing": int(df[col].isna().sum()),
                }
                for col in df.columns
            ],
        }

    def run_method(self, name: str, method: str) -> str:
        if method not in ALLOWED_METHODS:
            return f"'{method}' is not supported. Use one of: {', '.join(ALLOWED_METHODS)}."
        df = self.load(name)
        result = ALLOWED_METHODS[method](df)
        if isinstance(result, (pd.DataFrame, pd.Series)):
            with pd.option_context("display.max_columns", 50, "display.width", 160):
                return result.to_string()
        return str(result)


def infer_task_type(dtype: str, n_unique: int, n_rows: int) -> str:
    """Heuristic the LLM is expected to apply: categorical target -> classification."""
    numeric = any(t in dtype for t in ("int", "float"))
    if not numeric or dtype.startswith("bool"):
        return "classification"
    if "int" in dtype and n_unique <= max(10, int(0.02 * n_rows)):
        return "classification"
    return "regression"


# Shared store used by the tools (one per process)
STORE = DatasetStore()


def set_data_dir(path: os.PathLike | str) -> DatasetStore:
    """Point the shared store at another folder (clears the cache)."""
    STORE.data_dir = Path(path).resolve()
    STORE.clear()
    return STORE
