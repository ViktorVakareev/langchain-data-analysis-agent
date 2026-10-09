"""``OfflineDataModel`` - a deterministic, rule-based stand-in for an LLM.

It is **not an LLM**. It speaks the same tool-calling protocol as a real chat model
(emits ``AIMessage.tool_calls``, reads ``ToolMessage`` results), so the real
LangChain agent loop and the real tools run without an API key or any cost.
Use it for demos, CI and screenshots. Its "reasoning" is a fixed plan per intent:

- list      -> list_csv_files
- summary   -> list_csv_files -> get_dataset_summaries
- inspect   -> [list_csv_files] -> call_dataframe_method (head/describe/info/...)
- model     -> [list_csv_files] -> get_dataset_summaries -> evaluate_* per dataset,
               choosing the target column and the task type the way the lab
               expects an LLM to: from the column dtype and number of unique values.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .datasets import ALLOWED_METHODS, infer_task_type

TARGET_HINTS = ("target", "label", "class", "churn", "churned", "price", "price_k", "outcome", "y", "sales")

_INTENTS = [
    ("model", r"classif|regress|train|evaluat|model|predict|accuracy|machine learning|\bml\b"),
    ("summary", r"summar|structure|overview|about|tell me"),
    ("inspect", r"\b(" + "|".join(ALLOWED_METHODS) + r")\b|first rows|statistic|missing values|correlat"),
    ("list", r"\blist\b|available|which (files|datasets|data)|what (files|datasets|data)"),
]
_METHOD_ALIASES = {"first rows": "head", "statistic": "describe", "missing values": "missing", "correlat": "corr"}


def detect_intent(query: str) -> str:
    for intent, pattern in _INTENTS:
        if re.search(pattern, query, re.IGNORECASE):
            return intent
    return "unknown"


def detect_method(query: str) -> str:
    for method in ALLOWED_METHODS:
        if re.search(rf"\b{method}\b", query, re.IGNORECASE):
            return method
    for alias, method in _METHOD_ALIASES.items():
        if alias in query.lower():
            return method
    return "head"


def choose_target(summary: dict[str, Any], query: str = "") -> Optional[str]:
    names = [c["name"] for c in summary.get("columns", [])]
    for name in names:  # a column named in the question wins
        if re.search(rf"\b{re.escape(name)}\b", query, re.IGNORECASE):
            return name
    for hint in TARGET_HINTS:
        for name in names:
            if name.lower() == hint:
                return name
    return names[-1] if names else None


def _call(name: str, **args: Any) -> AIMessage:
    call = {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


def _parse(content: Any) -> Any:
    """Tool results arrive as JSON (create_agent) or Python reprs (classic AgentExecutor)."""
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(content)
        except (TypeError, ValueError, SyntaxError):
            continue
    return content


class OfflineDataModel(BaseChatModel):
    """Rule-based chat model that drives the data tools through real tool calls."""

    @property
    def _llm_type(self) -> str:
        return "offline-data-rules"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "OfflineDataModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        query = str(messages[start].content)
        # Map tool_call_id -> tool name (classic AgentExecutor leaves ToolMessage.name empty)
        names = {c["id"]: c["name"] for m in messages[start:] if isinstance(m, AIMessage) for c in m.tool_calls}
        results: dict[str, list[Any]] = {}
        for m in messages[start:]:
            if isinstance(m, ToolMessage):
                results.setdefault(m.name or names.get(m.tool_call_id, ""), []).append(_parse(m.content))
        message = self._next_step(query, results)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ------------------------------------------------------------------ planning
    def _next_step(self, query: str, results: dict[str, list[Any]]) -> AIMessage:
        intent = detect_intent(query)
        if intent == "unknown":
            return AIMessage(content=(
                "Offline demo model: I can list datasets, summarize them, run inspections "
                "(head, describe, info, missing, corr, ...) and train/evaluate a model. "
                "Try: 'Which datasets are available?' or 'Train a model on each dataset'."))

        named = re.findall(r"[\w\-.]+\.csv", query)
        if not named and "list_csv_files" not in results:
            return _call("list_csv_files")
        files = named or (results.get("list_csv_files", [None])[0] or [])
        if not files:
            return AIMessage(content="I couldn't find any CSV files in the data folder.")

        if intent == "list":
            return AIMessage(content="Available datasets:\n" + "\n".join(f"- {f}" for f in files))

        if intent == "inspect":
            method = detect_method(query)
            done = len(results.get("call_dataframe_method", []))
            if done < len(files):
                return _call("call_dataframe_method", file_name=files[done], method=method)
            outputs = results["call_dataframe_method"]
            return AIMessage(content="\n\n".join(f"{f} - {method}():\n{o}" for f, o in zip(files, outputs)))

        if "get_dataset_summaries" not in results:
            return _call("get_dataset_summaries", dataset_paths=list(files))
        summaries = [s for s in results["get_dataset_summaries"][0] if "error" not in s]

        if intent == "summary":
            return AIMessage(content=self._describe(summaries))

        # intent == "model": one evaluation per dataset, in order
        evaluations = results.get("evaluate_classification_dataset", []) + results.get("evaluate_regression_dataset", [])
        if len(evaluations) < len(summaries):
            summary = summaries[len(evaluations)]
            target = choose_target(summary, query)
            column = next(c for c in summary["columns"] if c["name"] == target)
            task = infer_task_type(column["dtype"], column["n_unique"], summary["rows"])
            tool_name = f"evaluate_{task}_dataset"
            return _call(tool_name, file_name=summary["file_name"], target_column=target)
        return AIMessage(content=self._report(evaluations))

    # ------------------------------------------------------------------ answers
    @staticmethod
    def _describe(summaries: list[dict[str, Any]]) -> str:
        lines = []
        for s in summaries:
            cols = ", ".join(f"{c['name']} ({c['dtype']})" for c in s["columns"])
            gaps = [f"{c['name']}: {c['missing']}" for c in s["columns"] if c["missing"]]
            lines.append(f"{s['file_name']}: {s['rows']} rows, {len(s['columns'])} columns - {cols}."
                         + (f" Missing values -> {', '.join(gaps)}." if gaps else ""))
        return "\n".join(lines)

    @staticmethod
    def _report(evaluations: list[dict[str, Any]]) -> str:
        lines = []
        for e in evaluations:
            if "error" in e:
                lines.append(f"- Error: {e['error']}")
            elif e["task"] == "classification":
                lines.append(f"- '{e['target_column']}' is categorical -> classification. "
                             f"Accuracy {e['accuracy']:.1%} (baseline {e['baseline_accuracy']:.1%}), "
                             f"F1 {e['f1_macro']:.2f}. Top features: {', '.join(e['top_features'])}.")
            else:
                lines.append(f"- '{e['target_column']}' is continuous -> regression. "
                             f"R² {e['r2_score']:.3f}, MAE {e['mean_absolute_error']:.2f}. "
                             f"Top features: {', '.join(e['top_features'])}.")
        return "Results:\n" + "\n".join(lines)
