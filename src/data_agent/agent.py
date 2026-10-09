"""Agent factory, provider switch and response helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage

from .tools import TOOLS

SYSTEM_PROMPT = (
    "You are DataWizard, a data science assistant for non-technical users. "
    "Use the available tools to analyze the CSV files - never invent numbers. "
    "Typical workflow: list the files, summarize their structure, inspect them if needed, "
    "then decide for each dataset whether the target is a classification problem "
    "(text/boolean or few distinct values) or a regression problem (continuous numbers) "
    "and call the matching evaluation tool. "
    "Explain results in plain language: what the metric means and whether it is good."
)

PROVIDERS = ("gemini", "openai", "offline")
DEFAULT_MODELS = {"gemini": "gemini-2.5-flash", "openai": "gpt-4o-mini"}


def get_llm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs: Any) -> BaseChatModel:
    """Build the chat model. ``provider`` defaults to $LLM_PROVIDER, then "gemini".

    - gemini:  free API key from Google AI Studio in $GOOGLE_API_KEY ($GEMINI_MODEL)
    - openai:  $OPENAI_API_KEY ($OPENAI_MODEL), billed per token
    - offline: rule-based stand-in, no key, no network, no cost
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model or os.getenv("GEMINI_MODEL", DEFAULT_MODELS["gemini"]),
                                      temperature=0, **kwargs)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", DEFAULT_MODELS["openai"]),
                          temperature=0, **kwargs)
    if provider == "offline":
        from .offline import OfflineDataModel

        return OfflineDataModel()
    raise ValueError(f"Unknown provider {provider!r}. Choose one of {PROVIDERS}.")


def build_agent(llm: Optional[BaseChatModel] = None, tools: Optional[Sequence] = None,
                system_prompt: str = SYSTEM_PROMPT):
    """Create the tool-calling agent: a LangGraph loop of model -> tools -> model ..."""
    return create_agent(model=llm or get_llm(), tools=list(tools or TOOLS), system_prompt=system_prompt)


def ask(agent, question: str, history: Optional[list] = None) -> dict:
    """Run one user turn. Pass the previous ``response["messages"]`` as history to keep context."""
    return agent.invoke({"messages": [*(history or []), ("human", question)]})


def final_answer(response: dict) -> str:
    content = response["messages"][-1].content
    if isinstance(content, list):  # some providers return content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def tool_calls(response: dict, since_last_human: bool = True) -> list[dict]:
    """Tool calls made in the response: [{"name", "args", "result"}, ...]."""
    messages = response["messages"]
    if since_last_human:
        last_human = max(i for i, m in enumerate(messages) if m.type == "human")
        messages = messages[last_human:]
    results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    calls = []
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            tool_msg = results.get(call["id"])
            calls.append({"name": call["name"], "args": call["args"],
                          "result": _parse(tool_msg.content) if tool_msg else None})
    return calls


def _parse(content: Any) -> Any:
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content
