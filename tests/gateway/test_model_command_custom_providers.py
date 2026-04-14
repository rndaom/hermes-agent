"""Regression tests for gateway /model support of config.yaml custom_providers."""

import yaml
import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    return runner


def _make_event(text="/model"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"
        ),
    )


class _PickerAdapter:
    def __init__(self):
        self.calls = []

    async def send_model_picker(self, **kwargs):
        self.calls.append(kwargs)
        return type("Result", (), {"success": True})()


@pytest.mark.asyncio
async def test_handle_model_command_lists_saved_custom_provider(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5.4",
                    "provider": "openai-codex",
                    "base_url": "https://chatgpt.com/backend-api/codex",
                },
                "providers": {},
                "custom_providers": [
                    {
                        "name": "Local (127.0.0.1:4141)",
                        "base_url": "http://127.0.0.1:4141/v1",
                        "model": "rotator-openrouter-coding",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    result = await _make_runner()._handle_model_command(_make_event())

    assert result is not None
    assert "Local (127.0.0.1:4141)" in result
    assert "custom:local-(127.0.0.1:4141)" in result
    assert "rotator-openrouter-coding" in result


@pytest.mark.asyncio
async def test_handle_model_command_sends_picker_for_providers_dict_entry_on_3ds(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "default": "gpt-5-mini",
                    "provider": "openai-direct-primary",
                    "base_url": "https://api.openai.com/v1",
                },
                "providers": {
                    "openai-direct-primary": {
                        "name": "OpenAI Direct (Primary)",
                        "api": "https://api.openai.com/v1",
                        "default_model": "gpt-5-mini",
                        "models": ["gpt-5-mini", "gpt-5.4"],
                        "transport": "codex_responses",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    picker = _PickerAdapter()
    runner = _make_runner()
    runner.adapters = {Platform.THREEDS: picker}

    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})

    event = MessageEvent(
        text="/model",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.THREEDS,
            chat_id="3ds:old3ds",
            chat_type="dm",
            thread_id="main",
        ),
        message_id="user-123",
    )

    result = await runner._handle_model_command(event)

    assert result is None
    assert len(picker.calls) == 1
    call = picker.calls[0]
    assert call["chat_id"] == "3ds:old3ds"
    assert call["current_model"] == "gpt-5-mini"
    assert call["current_provider"] == "openai-direct-primary"
    assert call["reply_to"] == "user-123"

    provider = next(
        p for p in call["providers"] if p["slug"] == "openai-direct-primary"
    )
    assert provider["name"] == "OpenAI Direct (Primary)"
    assert provider["models"] == ["gpt-5-mini", "gpt-5.4"]
    assert provider["is_current"] is True
