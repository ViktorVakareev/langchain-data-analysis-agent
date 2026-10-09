"""Agent orchestration: the real create_agent loop + real tools, with fake/offline models."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conftest import tool_call
from data_agent import ask, build_agent, final_answer, get_llm, tool_calls
from data_agent.offline import OfflineDataModel, choose_target, detect_intent, detect_method


def test_loop_runs_tools_and_returns_final_answer(scripted_model):
    llm = scripted_model(
        tool_call("list_csv_files", "c1"),
        tool_call("evaluate_regression_dataset", "c2", file_name="house-prices.csv", target_column="price_k"),
        AIMessage(content="R² is 0.88 - the model explains most of the price variation."),
    )
    response = ask(build_agent(llm), "How well can we predict house prices?")
    assert [type(m) for m in response["messages"]] == [HumanMessage, AIMessage, ToolMessage, AIMessage, ToolMessage, AIMessage]
    calls = tool_calls(response)
    assert calls[0]["result"] == ["customer-churn.csv", "house-prices.csv"]
    assert calls[1]["result"]["r2_score"] > 0.8
    assert final_answer(response).startswith("R² is 0.88")


def test_tool_errors_flow_back_to_the_model(scripted_model):
    llm = scripted_model(
        tool_call("evaluate_classification_dataset", file_name="customer-churn.csv", target_column="colour"),
        AIMessage(content="That column doesn't exist."),
    )
    response = ask(build_agent(llm), "Predict colour")
    assert "not found" in tool_calls(response)[0]["result"]["error"]


def test_history_keeps_context_and_tool_calls_are_per_turn(scripted_model):
    agent = build_agent(scripted_model(tool_call("list_csv_files"), AIMessage(content="2 files"), AIMessage(content="You asked about files.")))
    first = ask(agent, "List files")
    second = ask(agent, "What did I ask?", history=first["messages"])
    assert len(second["messages"]) == len(first["messages"]) + 2
    assert tool_calls(second) == [] and len(tool_calls(second, since_last_human=False)) == 1


def test_final_answer_handles_content_blocks():
    msg = AIMessage(content=[{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}])
    assert final_answer({"messages": [msg]}) == "Hello there"


# ---- offline rule-based model --------------------------------------------------------
@pytest.mark.parametrize(
    "query, intent",
    [("Which datasets are available?", "list"), ("Tell me about the data", "summary"),
     ("Summarize the columns", "summary"), ("Show describe for house-prices.csv", "inspect"),
     ("Train a model", "model"), ("Is it classification or regression?", "model"), ("hello", "unknown")],
)
def test_detect_intent(query, intent):
    assert detect_intent(query) == intent


@pytest.mark.parametrize("query, method", [("show the tail", "tail"), ("first rows please", "head"),
                                           ("any missing values?", "missing"), ("correlation", "corr"), ("show it", "head")])
def test_detect_method(query, method):
    assert detect_method(query) == method


def test_choose_target_prefers_named_column_then_hints_then_last():
    summary = {"columns": [{"name": "a"}, {"name": "price"}, {"name": "z"}]}
    assert choose_target(summary, "predict a") == "a"
    assert choose_target(summary) == "price"
    assert choose_target({"columns": [{"name": "a"}, {"name": "z"}]}) == "z"


@pytest.mark.parametrize(
    "query, expected_tools",
    [
        ("Which datasets are available?", ["list_csv_files"]),
        ("Tell me about the datasets", ["list_csv_files", "get_dataset_summaries"]),
        ("Show the head of house-prices.csv", ["call_dataframe_method"]),
        ("Show missing values", ["list_csv_files", "call_dataframe_method", "call_dataframe_method"]),
        ("Is each dataset classification or regression? Train a model for each.",
         ["list_csv_files", "get_dataset_summaries", "evaluate_classification_dataset", "evaluate_regression_dataset"]),
        ("Train a model on customer-churn.csv to predict contract",
         ["get_dataset_summaries", "evaluate_classification_dataset"]),
    ],
)
def test_offline_model_end_to_end(query, expected_tools):
    response = ask(build_agent(OfflineDataModel()), query)
    assert [c["name"] for c in tool_calls(response)] == expected_tools
    assert final_answer(response)


def test_offline_model_picks_target_and_task_like_an_llm_should():
    response = ask(build_agent(OfflineDataModel()), "Train and evaluate a model for each dataset")
    evals = {c["args"]["file_name"]: c for c in tool_calls(response) if c["name"].startswith("evaluate")}
    assert evals["customer-churn.csv"]["args"]["target_column"] == "churned"
    assert evals["house-prices.csv"]["args"]["target_column"] == "price_k"
    assert "classification" in final_answer(response) and "regression" in final_answer(response)


def test_offline_model_handles_empty_folder_and_unknown(data_dir):
    assert "Offline demo model" in final_answer(ask(build_agent(OfflineDataModel()), "hello"))
    for f in data_dir.glob("*.csv"):
        f.unlink()
    assert "couldn't find any CSV" in final_answer(ask(build_agent(OfflineDataModel()), "List the datasets"))


def test_get_llm_providers(monkeypatch):
    assert isinstance(get_llm("offline"), OfflineDataModel)
    with pytest.raises(ValueError):
        get_llm("nope")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    assert type(get_llm("gemini")).__name__ == "ChatGoogleGenerativeAI"
    assert type(get_llm("openai")).__name__ == "ChatOpenAI"
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    assert isinstance(get_llm(), OfflineDataModel)


def test_tool_schemas_convert_for_gemini_and_openai(monkeypatch):
    """Catches schemas a provider would reject - without making an API call."""
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_google_genai._function_utils import convert_to_genai_function_declarations

    from data_agent import TOOLS

    declared = [f.name for t in convert_to_genai_function_declarations(TOOLS) for f in (t.function_declarations or [])]
    assert declared == [t.name for t in TOOLS]
    assert all(convert_to_openai_tool(t)["function"]["name"] == t.name for t in TOOLS)
