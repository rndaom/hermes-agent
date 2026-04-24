import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.platforms.base import MessageEvent
from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE
from gateway.session import build_session_key
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.mark.asyncio
async def test_cancel_background_tasks_cancels_inflight_message_processing():
    _runner, adapter = make_restart_runner()
    release = asyncio.Event()

    async def block_forever(_event):
        await release.wait()
        return None

    adapter.set_message_handler(block_forever)
    event = MessageEvent(text="work", source=make_restart_source(), message_id="1")

    await adapter.handle_message(event)
    await asyncio.sleep(0)

    session_key = build_session_key(event.source)
    assert session_key in adapter._active_sessions
    assert adapter._background_tasks

    await adapter.cancel_background_tasks()

    assert adapter._background_tasks == set()
    assert adapter._active_sessions == {}
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_gateway_stop_interrupts_running_agents_and_cancels_adapter_tasks():
    runner, adapter = make_restart_runner()
    runner._pending_messages = {"session": "pending text"}
    runner._pending_approvals = {"session": {"command": "rm -rf /tmp/x"}}
    runner._restart_drain_timeout = 0.0

    release = asyncio.Event()

    async def block_forever(_event):
        await release.wait()
        return None

    adapter.set_message_handler(block_forever)
    event = MessageEvent(text="work", source=make_restart_source(), message_id="1")
    await adapter.handle_message(event)
    await asyncio.sleep(0)

    disconnect_mock = AsyncMock()
    adapter.disconnect = disconnect_mock

    session_key = build_session_key(event.source)
    running_agent = MagicMock()
    runner._running_agents = {session_key: running_agent}

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    running_agent.interrupt.assert_called_once_with("Gateway shutting down")
    disconnect_mock.assert_awaited_once()
    assert runner.adapters == {}
    assert runner._running_agents == {}
    assert runner._pending_messages == {}
    assert runner._pending_approvals == {}
    assert runner._shutdown_event.is_set() is True


@pytest.mark.asyncio
async def test_gateway_stop_drains_running_agents_before_disconnect():
    runner, adapter = make_restart_runner()
    disconnect_mock = AsyncMock()
    adapter.disconnect = disconnect_mock

    running_agent = MagicMock()
    runner._running_agents = {"session": running_agent}

    async def finish_agent():
        await asyncio.sleep(0.05)
        runner._running_agents.clear()

    asyncio.create_task(finish_agent())

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    running_agent.interrupt.assert_not_called()
    disconnect_mock.assert_awaited_once()
    assert runner._shutdown_event.is_set() is True


@pytest.mark.asyncio
async def test_gateway_stop_interrupts_after_drain_timeout():
    runner, adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.05

    disconnect_mock = AsyncMock()
    adapter.disconnect = disconnect_mock

    running_agent = MagicMock()
    runner._running_agents = {"session": running_agent}

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    running_agent.interrupt.assert_called_once_with("Gateway shutting down")
    disconnect_mock.assert_awaited_once()
    assert runner._shutdown_event.is_set() is True


def test_describe_draining_agents_includes_session_state_and_age(monkeypatch):
    runner, _adapter = make_restart_runner()
    monkeypatch.setattr(gateway_run.time, "time", lambda: 200.0)

    running_agent = MagicMock()
    running_agent.session_id = "sess-running"
    runner._running_agents = {
        "discord:chat:thread": running_agent,
        "telegram:pending": gateway_run._AGENT_PENDING_SENTINEL,
    }
    runner._running_agents_ts = {
        "discord:chat:thread": 188.2,
        "telegram:pending": 195.0,
    }

    summary = runner._describe_draining_agents()

    assert "discord:chat:thread" in summary
    assert "state=running" in summary
    assert "session_id=sess-running" in summary
    assert "age=12s" in summary
    assert "telegram:pending" in summary
    assert "state=pending" in summary
    assert "age=5s" in summary


@pytest.mark.asyncio
async def test_gateway_stop_logs_blocking_session_details_after_drain_timeout(monkeypatch):
    runner, adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.0
    adapter.disconnect = AsyncMock()

    running_agent = MagicMock()
    running_agent.session_id = "sess-1"
    runner._running_agents = {"discord:123:456": running_agent}
    runner._running_agents_ts = {"discord:123:456": 90.0}
    monkeypatch.setattr(gateway_run.time, "time", lambda: 100.0)

    with (
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.write_runtime_status"),
        patch("gateway.run.logger") as mock_logger,
    ):
        await runner.stop()

    warning_call = mock_logger.warning.call_args_list[0]
    assert "Gateway drain timed out after %.1fs with %d active agent(s): %s; interrupting remaining work." in warning_call.args[0]
    assert warning_call.args[1] == 0.0
    assert warning_call.args[2] == 1
    assert "discord:123:456" in warning_call.args[3]
    assert "state=running" in warning_call.args[3]
    assert "session_id=sess-1" in warning_call.args[3]
    assert "age=10s" in warning_call.args[3]


@pytest.mark.asyncio
async def test_gateway_stop_service_restart_sets_named_exit_code():
    runner, adapter = make_restart_runner()
    adapter.disconnect = AsyncMock()

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop(restart=True, service_restart=True)

    assert runner._exit_code == GATEWAY_SERVICE_RESTART_EXIT_CODE


@pytest.mark.asyncio
async def test_drain_active_agents_throttles_status_updates():
    runner, _adapter = make_restart_runner()
    runner._update_runtime_status = MagicMock()

    runner._running_agents = {"a": MagicMock(), "b": MagicMock()}

    async def finish_agents():
        await asyncio.sleep(0.12)
        runner._running_agents.pop("a")
        await asyncio.sleep(0.12)
        runner._running_agents.clear()

    task = asyncio.create_task(finish_agents())
    await runner._drain_active_agents(1.0)
    await task

    # Start, one count-change update, and final update. Allow one extra update
    # if the loop observes the zero-agent state before exiting.
    assert 3 <= runner._update_runtime_status.call_count <= 4
