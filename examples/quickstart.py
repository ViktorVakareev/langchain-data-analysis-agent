"""Quickstart: ask DataWizard a few questions and show which tools it used.

    python examples/quickstart.py            # provider from .env (default: gemini)
    python examples/quickstart.py offline    # no API key, no cost
"""

import sys

from dotenv import load_dotenv

from data_agent import ask, build_agent, final_answer, get_llm, tool_calls

load_dotenv()
agent = build_agent(get_llm(sys.argv[1] if len(sys.argv) > 1 else None))

for question in [
    "Which datasets are available?",
    "Is each dataset a classification or a regression problem? Train and evaluate a model for each.",
]:
    response = ask(agent, question)
    print(f"\nQ: {question}")
    for call in tool_calls(response):
        print(f"   🔧 {call['name']}({call['args']})")
    print(f"A: {final_answer(response)}")
