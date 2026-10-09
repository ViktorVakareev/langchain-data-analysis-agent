"""Live evaluation with a real LLM - skipped unless an API key is set.

    pytest -m live -v                 # uses Gemini if GOOGLE_API_KEY is set, OpenAI if OPENAI_API_KEY is set

Asserts on WHICH tools were called with WHAT arguments, not on the model's wording.
"""

import os

import pytest

from data_agent import ask, build_agent, get_llm, tool_calls

PROVIDERS = [
    pytest.param("gemini", marks=pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")),
    pytest.param("openai", marks=pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")),
]
pytestmark = pytest.mark.live


@pytest.mark.parametrize("provider", PROVIDERS)
def test_lists_files(provider):
    names = [c["name"] for c in tool_calls(ask(build_agent(get_llm(provider)), "Which datasets are available?"))]
    assert "list_csv_files" in names


@pytest.mark.parametrize("provider", PROVIDERS)
def test_picks_the_right_task_and_target(provider):
    response = ask(build_agent(get_llm(provider)),
                   "For each dataset, decide if it is classification or regression, then train and evaluate a model.")
    evals = {(c["name"], c["args"].get("file_name"), c["args"].get("target_column")) for c in tool_calls(response)}
    assert ("evaluate_classification_dataset", "customer-churn.csv", "churned") in evals
    assert ("evaluate_regression_dataset", "house-prices.csv", "price_k") in evals
