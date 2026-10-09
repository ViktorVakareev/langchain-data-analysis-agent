"""Generate and execute notebooks/walkthrough.ipynb (offline provider by default).

    python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# DataWizard — walkthrough\n\n"
       "An AI agent that analyzes CSV files through **LangChain tools**. Runs **offline by default** "
       "(rule-based stand-in model: no API key, no cost). Set `PROVIDER = \"gemini\"` (free key) or `\"openai\"` for a real LLM."),
    code("PROVIDER = \"offline\"  # \"gemini\" | \"openai\" | \"offline\"\n\n"
         "import os, json\nfrom pathlib import Path\nfrom dotenv import load_dotenv\nload_dotenv()\n\n"
         "from data_agent import (TOOLS, set_data_dir, list_csv_files, get_dataset_summaries, call_dataframe_method,\n"
         "                        evaluate_classification_dataset, evaluate_regression_dataset,\n"
         "                        build_agent, get_llm, ask, final_answer, tool_calls)\n"
         "from data_agent.cli import print_trace\n"
         "set_data_dir(Path.cwd().parent / \"data\" if Path.cwd().name == \"notebooks\" else \"data\")"),
    md("## 1. The tools — what the LLM sees\nName + docstring + argument schema. The model picks tools from these descriptions alone."),
    code("for t in TOOLS:\n    print(f\"{t.name:32} args={list(t.args)}\")"),
    md("## 2. Call the tools directly (no LLM)\nThe agent passes **file names**, not data — the DataFrames stay cached in memory."),
    code("print(list_csv_files.invoke({}))\n"
         "summary = get_dataset_summaries.invoke({\"dataset_paths\": [\"customer-churn.csv\"]})[0]\n"
         "for c in summary[\"columns\"]:\n    print(f\"{c['name']:18} {c['dtype']:8} n_unique={c['n_unique']}\")"),
    code("print(call_dataframe_method.invoke({\"file_name\": \"house-prices.csv\", \"method\": \"head\"}))"),
    code("print(json.dumps(evaluate_classification_dataset.invoke({\"file_name\": \"customer-churn.csv\", \"target_column\": \"churned\"}), indent=1))\n"
         "print(json.dumps(evaluate_regression_dataset.invoke({\"file_name\": \"house-prices.csv\", \"target_column\": \"price_k\"}), indent=1))"),
    md("## 3. One step vs. the full loop\n"
       "A model on its own decides **one** action. The agent (the lab's `AgentExecutor`, here `create_agent`) "
       "keeps looping — act, observe, decide — until it can answer. `stream` shows each step."),
    code("agent = build_agent(get_llm(PROVIDER))\n"
         "for step in agent.stream({\"messages\": [(\"human\", \"Tell me about the datasets\")]}, stream_mode=\"updates\"):\n"
         "    for node, update in step.items():\n"
         "        msg = update[\"messages\"][-1]\n"
         "        what = [c[\"name\"] for c in msg.tool_calls] if getattr(msg, \"tool_calls\", None) else str(msg.content)[:90]\n"
         "        print(f\"{node:6} → {what}\")"),
    md("## 4. The DataWizard question: classification or regression?"),
    code("response = ask(agent, \"Is each dataset a classification or a regression problem? Train and evaluate a model for each.\")\n"
         "print_trace(response[\"messages\"])"),
    md("## 5. Evaluate the agent: check the tool calls, not the wording"),
    code("expected = {(\"evaluate_classification_dataset\", \"customer-churn.csv\", \"churned\"),\n"
         "            (\"evaluate_regression_dataset\", \"house-prices.csv\", \"price_k\")}\n"
         "actual = {(c[\"name\"], c[\"args\"].get(\"file_name\"), c[\"args\"].get(\"target_column\")) for c in tool_calls(response)}\n"
         "for item in sorted(expected):\n    print(\"✅\" if item in actual else \"❌\", item)\n"
         "print(f\"\\n{len(expected & actual)}/{len(expected)} checks passed\")"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
nbf.write(nb, ROOT / "notebooks" / "walkthrough.ipynb")
print("wrote notebooks/walkthrough.ipynb")
