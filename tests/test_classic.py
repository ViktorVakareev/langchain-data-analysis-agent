"""The lab's original pattern (langchain-classic): one-step agent vs AgentExecutor."""

import pytest

classic = pytest.importorskip("langchain_classic.agents")
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402

from data_agent import TOOLS  # noqa: E402
from data_agent.offline import OfflineDataModel  # noqa: E402

PROMPT = ChatPromptTemplate.from_messages(
    [("system", "You are a data science assistant."), ("user", "{input}"), ("placeholder", "{agent_scratchpad}")])


def test_raw_agent_takes_only_one_step():
    agent = classic.create_tool_calling_agent(OfflineDataModel(), TOOLS, PROMPT)
    step = agent.invoke({"input": "Can you tell me about the datasets?", "intermediate_steps": []})
    assert step[0].tool == "list_csv_files"


def test_agent_executor_runs_the_full_loop():
    agent = classic.create_tool_calling_agent(OfflineDataModel(), TOOLS, PROMPT)
    executor = classic.AgentExecutor(agent=agent, tools=TOOLS, return_intermediate_steps=True)
    result = executor.invoke({"input": "Train and evaluate a model for each dataset"})
    assert [a.tool for a, _ in result["intermediate_steps"]] == [
        "list_csv_files", "get_dataset_summaries", "evaluate_classification_dataset", "evaluate_regression_dataset"]
    assert "regression" in result["output"]
