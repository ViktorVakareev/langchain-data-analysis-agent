"""DataWizard - an AI data analysis agent built with LangChain tool calling."""

from .agent import SYSTEM_PROMPT, ask, build_agent, final_answer, get_llm, tool_calls
from .datasets import STORE, DatasetStore, infer_task_type, set_data_dir
from .modeling import evaluate_classification, evaluate_regression
from .tools import (
    TOOLS,
    call_dataframe_method,
    evaluate_classification_dataset,
    evaluate_regression_dataset,
    get_dataset_summaries,
    list_csv_files,
    preload_datasets,
)

__version__ = "1.0.0"

__all__ = [
    "STORE", "SYSTEM_PROMPT", "TOOLS", "DatasetStore", "ask", "build_agent", "call_dataframe_method",
    "evaluate_classification", "evaluate_classification_dataset", "evaluate_regression",
    "evaluate_regression_dataset", "final_answer", "get_dataset_summaries", "get_llm",
    "infer_task_type", "list_csv_files", "preload_datasets", "set_data_dir", "tool_calls",
]
