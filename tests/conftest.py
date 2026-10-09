import shutil
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from data_agent.datasets import set_data_dir

REPO_DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(autouse=True)
def data_dir(tmp_path):
    """Every test gets a fresh copy of the sample datasets and an empty cache."""
    for csv in REPO_DATA.glob("*.csv"):
        shutil.copy(csv, tmp_path / csv.name)
    set_data_dir(tmp_path)
    yield tmp_path
    set_data_dir(REPO_DATA)


class ScriptedToolModel(GenericFakeChatModel):
    """Fake chat model replaying scripted AIMessages (incl. tool calls)."""

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name, call_id="call_1", **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def scripted_model():
    return lambda *messages: ScriptedToolModel(messages=iter(messages))
