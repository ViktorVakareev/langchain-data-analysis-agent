"""The lab's original pattern: a one-step agent + AgentExecutor (LangChain "classic" API).

The lab used LangChain 0.3's create_openai_tools_agent + AgentExecutor. In LangChain 1.x
these live in `langchain-classic`; create_tool_calling_agent is the provider-neutral
version. This script shows the two ideas the lab teaches:

1. A raw agent decides only ONE step (here: "call list_csv_files") and stops.
2. AgentExecutor runs the full ReAct loop: think -> act -> observe -> ... -> answer.

    pip install -e ".[classic]"
    python examples/classic_agent_executor.py offline     # or gemini / openai
"""

import sys

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from data_agent import TOOLS, get_llm

load_dotenv()
llm = get_llm(sys.argv[1] if len(sys.argv) > 1 else None)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a data science assistant. Use the available tools to analyze CSV files. "
               "Your job is to determine whether each dataset is for classification or regression, "
               "based on its structure."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # where intermediate tool calls/results go
])
agent = create_tool_calling_agent(llm, TOOLS, prompt)

# 1) Raw agent: a single reasoning step
step = agent.invoke({"input": "Can you tell me about the datasets?", "intermediate_steps": []})
action = step[0]
print("🧠 Raw agent decided ONE step only:")
print("   tool:", action.tool, "| input:", action.tool_input)

# 2) AgentExecutor: the full loop until a final answer
executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True, handle_parsing_errors=True)
result = executor.invoke({"input": "Is each dataset a classification or a regression problem? Train and evaluate a model for each."})
print("\n🤖", result["output"])
