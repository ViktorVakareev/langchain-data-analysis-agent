"""DataWizard command line: ask questions about CSV files in plain English.

    data-agent "Which datasets are available?" --provider offline --trace
    data-agent --provider gemini            # interactive chat (type 'exit' to quit)
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .agent import PROVIDERS, ask, build_agent, final_answer, get_llm
from .datasets import set_data_dir

_COLOR = sys.stdout.isatty() or bool(os.getenv("FORCE_COLOR"))


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _short(text: str, limit: int = 320) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f" … (+{len(text) - limit} chars)"


def print_trace(messages: list, start: int = 0) -> None:
    """Print each step of the agent loop: question, tool calls, tool results, answer."""
    for i, msg in enumerate(messages[start:], start=start + 1):
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, HumanMessage):
            print(f"{_c('1;36', f'[{i}] Human')}       {msg.content}")
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"{_c('1;33', f'[{i}] AI → tool')}   {_c('33', call['name'])}({call['args']})")
        elif isinstance(msg, ToolMessage):
            body = textwrap.indent(_short(msg.content), " " * 16).lstrip()
            print(f"{_c('1;35', f'[{i}] Tool')}        {_c('35', msg.name)} → {body}")
        elif isinstance(msg, AIMessage):
            print(f"{_c('1;32', f'[{i}] AI answer')}   {final_answer({'messages': [msg]})}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="data-agent", description="DataWizard - AI data analysis agent (LangChain tool calling)")
    parser.add_argument("question", nargs="*", help="question to ask; omit for interactive chat")
    parser.add_argument("-p", "--provider", choices=PROVIDERS, default=os.getenv("LLM_PROVIDER", "gemini"))
    parser.add_argument("-m", "--model", default=None, help="override the model name")
    parser.add_argument("-d", "--data-dir", default=os.getenv("DATA_DIR", "data"), help="folder with CSV files")
    parser.add_argument("-t", "--trace", action="store_true", help="show every tool call and result")
    args = parser.parse_args(argv)

    set_data_dir(args.data_dir)
    agent = build_agent(get_llm(args.provider, args.model))
    history: list = []

    def run(question: str) -> None:
        nonlocal history
        response = ask(agent, question, history)
        if args.trace:
            print_trace(response["messages"], start=len(history))
        else:
            print(_c("1;32", "DataWizard: ") + final_answer(response))
        history = response["messages"]

    if args.question:
        run(" ".join(args.question))
        return 0

    print(f"📊 DataWizard {_c('2', f'(provider: {args.provider}, data: {args.data_dir})')} - type 'exit' to quit")
    while True:
        try:
            question = input(_c("1;36", "\nYou: ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit", ""}:
            break
        run(question)
    print("see ya later")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
