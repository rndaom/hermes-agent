import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

from run_agent import AIAgent


def _make_tool_defs(*names):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("memory", "web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://test.example.com/v1",
            provider="openrouter",
            api_mode="chat_completions",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a


def test_compression_flush_flag_defaults_off_from_config():
    with (
        patch("hermes_cli.config.load_config", return_value={"memory": {"flush_on_compress": False}}),
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://test.example.com/v1",
            provider="openrouter",
            api_mode="chat_completions",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert a._compression_memory_flush_enabled is False


def test_non_dict_config_falls_back_to_safe_defaults():
    with (
        patch("hermes_cli.config.load_config", return_value=[]),
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://test.example.com/v1",
            provider="openrouter",
            api_mode="chat_completions",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert a._compression_memory_flush_enabled is False


def test_memory_store_receives_project_memory_config():
    with (
        patch(
            "hermes_cli.config.load_config",
            return_value={
                "memory": {
                    "memory_enabled": True,
                    "user_profile_enabled": True,
                    "project_memory_char_limit": 321,
                    "project_memory_ttl_days": 14,
                }
            },
        ),
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("tools.memory_tool.MemoryStore") as mock_store,
        patch.dict(os.environ, {"TERMINAL_CWD": "/tmp/project-memory-test"}, clear=False),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://test.example.com/v1",
            provider="openrouter",
            api_mode="chat_completions",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert a._memory_store is mock_store.return_value
    kwargs = mock_store.call_args.kwargs
    assert kwargs["project_memory_char_limit"] == 321
    assert kwargs["project_memory_ttl_days"] == 14
    assert kwargs["working_dir"] == "/tmp/project-memory-test"


def test_invoke_memory_tool_forwards_scope_ttl_and_working_dir(agent):
    agent._memory_store = MagicMock()
    with (
        patch("run_agent.get_active_env", return_value=SimpleNamespace(cwd="/tmp/repo-b")),
        patch("tools.memory_tool.memory_tool", return_value='{"success": true}') as mock_memory,
    ):
        agent._invoke_tool(
            "memory",
            {
                "action": "add",
                "target": "memory",
                "content": "project note",
                "scope": "project",
                "ttl_days": 7,
            },
            effective_task_id="task-1",
        )

    agent._memory_store.refresh_project_context.assert_called_once_with("/tmp/repo-b")
    kwargs = mock_memory.call_args.kwargs
    assert kwargs["scope"] == "project"
    assert kwargs["ttl_days"] == 7
    assert kwargs["working_dir"] == "/tmp/repo-b"


def test_project_scoped_memory_does_not_mirror_to_external_provider(agent):
    agent._memory_store = MagicMock()
    agent._memory_manager = MagicMock()
    with patch("tools.memory_tool.memory_tool", return_value='{"success": true}'):
        agent._invoke_tool(
            "memory",
            {
                "action": "add",
                "target": "memory",
                "content": "project note",
                "scope": "PROJECT",
            },
            effective_task_id="task-1",
        )

    agent._memory_manager.on_memory_write.assert_not_called()


def test_sync_dynamic_session_context_rebuilds_prompt_after_repo_change(agent):
    agent._memory_store = MagicMock()
    agent._memory_store.refresh_project_context.return_value = True
    agent._cached_system_prompt = "old system prompt"
    agent._build_system_prompt = MagicMock(return_value="new system prompt")
    agent._session_db = None
    agent._last_context_working_dir = "/tmp/repo-a"

    with patch("run_agent.get_active_env", return_value=SimpleNamespace(cwd="/tmp/repo-b")):
        changed = agent._sync_dynamic_session_context("task-1", system_message="system prompt")

    assert changed is True
    agent._memory_store.refresh_project_context.assert_called_once_with("/tmp/repo-b")
    agent._build_system_prompt.assert_called_once_with("system prompt")
    assert agent._cached_system_prompt == "new system prompt"
    assert agent._last_context_working_dir == "/tmp/repo-b"


def test_sync_dynamic_session_context_skips_rebuild_when_cwd_is_unchanged(agent):
    agent._memory_store = MagicMock()
    agent._memory_store.refresh_project_context.return_value = False
    agent._cached_system_prompt = "old system prompt"
    agent._build_system_prompt = MagicMock(return_value="new system prompt")
    agent._session_db = None
    agent._last_context_working_dir = "/tmp/repo-a"

    with patch("run_agent.get_active_env", return_value=SimpleNamespace(cwd="/tmp/repo-a")):
        changed = agent._sync_dynamic_session_context("task-1", system_message="system prompt")

    assert changed is False
    agent._memory_store.refresh_project_context.assert_called_once_with("/tmp/repo-a")
    agent._build_system_prompt.assert_not_called()
    assert agent._cached_system_prompt == "old system prompt"
