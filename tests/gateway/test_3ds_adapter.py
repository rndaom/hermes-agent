import pytest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


def _create_app(adapter):
    app = web.Application()
    app.router.add_get("/api/v2/health", adapter._handle_v2_health)
    app.router.add_get("/api/v2/capabilities", adapter._handle_capabilities)
    app.router.add_post("/api/v2/messages", adapter._handle_messages)
    app.router.add_get("/api/v2/events", adapter._handle_events)
    app.router.add_post("/api/v2/interactions/{request_id}/respond", adapter._handle_interaction_response)
    return app


def test_platform_enum_has_3ds():
    assert Platform.THREEDS.value == "3ds"


def test_tools_config_and_prompt_hint_include_3ds():
    from hermes_cli.tools_config import PLATFORMS
    from hermes_cli.skills_config import PLATFORMS as SKILLS_PLATFORMS
    from agent.prompt_builder import PLATFORM_HINTS
    from toolsets import TOOLSETS

    assert "3ds" in PLATFORMS
    assert PLATFORMS["3ds"]["default_toolset"] == "hermes-3ds"
    assert "3ds" in SKILLS_PLATFORMS
    assert "3ds" in PLATFORM_HINTS
    assert "hermes-3ds" in TOOLSETS
    assert "hermes-3ds" in TOOLSETS["hermes-gateway"]["includes"]


def test_load_gateway_config_env_override_enables_3ds(monkeypatch):
    monkeypatch.setenv("THREEDS_ENABLED", "true")
    monkeypatch.setenv("THREEDS_HOST", "0.0.0.0")
    monkeypatch.setenv("THREEDS_PORT", "8787")
    monkeypatch.setenv("THREEDS_AUTH_TOKEN", "3ds-dev-token")
    monkeypatch.setenv("THREEDS_DEVICE_ID", "old3ds")

    from gateway.config import load_gateway_config

    config = load_gateway_config()
    assert Platform.THREEDS in config.platforms
    platform_cfg = config.platforms[Platform.THREEDS]
    assert platform_cfg.enabled is True
    assert platform_cfg.extra["host"] == "0.0.0.0"
    assert platform_cfg.extra["port"] == 8787
    assert platform_cfg.extra["auth_token"] == "3ds-dev-token"
    assert platform_cfg.extra["device_id"] == "old3ds"


def test_gateway_config_connected_platforms_includes_3ds():
    config = GatewayConfig(platforms={Platform.THREEDS: PlatformConfig(enabled=True)})
    assert Platform.THREEDS in config.get_connected_platforms()


def test_runner_authorizes_3ds_without_user_allowlist():
    runner = GatewayRunner(GatewayConfig())
    source = type("Source", (), {"platform": Platform.THREEDS, "user_id": "old3ds"})()
    assert runner._is_user_authorized(source) is True


def test_update_command_allowed_platforms_include_3ds():
    assert Platform.THREEDS in GatewayRunner._UPDATE_ALLOWED_PLATFORMS


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_native_health_and_capabilities_endpoints_return_expected_payload():
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "tok"}))
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/v1/health")
        assert resp.status == 404

        resp = await cli.get("/api/v2/health")
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True
        assert body["platform"] == "3ds"
        assert body["version"]

        resp = await cli.get("/api/v2/capabilities?token=tok")
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True
        assert body["transport"] == "http-long-poll"


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_messages_ack_returns_cursor_and_events_return_reply_to():
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "tok"}))
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/v2/messages",
            json={
                "token": "tok",
                "device_id": "old3ds",
                "conversation_id": "main",
                "text": "hello",
            },
        )
        assert resp.status == 200
        ack = await resp.json()
        assert ack["ok"] is True
        assert ack["chat_id"] == "3ds:old3ds"
        assert ack["conversation_id"] == "main"
        assert "message_id" in ack
        assert "cursor" in ack

        await adapter.send(
            chat_id="3ds:old3ds",
            content="Hi from Hermes",
            reply_to=ack["message_id"],
            metadata={"thread_id": "main"},
        )

        resp = await cli.get(
            f"/api/v2/events?token=tok&device_id=old3ds&conversation_id=main&cursor={ack['cursor']}&wait=1"
        )
        assert resp.status == 200
        event_body = await resp.json()
        assert event_body["ok"] is True
        assert event_body["cursor"] > ack["cursor"]
        assert event_body["event"]["type"] == "message.created"
        assert event_body["event"]["text"] == "Hi from Hermes"
        assert event_body["event"]["reply_to"] == ack["message_id"]


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_interaction_response_resolves_gateway_approval():
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "tok"}))
    await adapter.send_exec_approval(
        chat_id="3ds:old3ds",
        command="rm -rf /tmp/demo",
        session_key="agent:main:3ds:dm:3ds:old3ds:main",
        description="dangerous command",
        metadata={"thread_id": "main"},
    )
    request_id = next(iter(adapter._pending_interactions))
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        with patch("gateway.platforms.threeds.resolve_gateway_approval", return_value=1) as mock_resolve:
            resp = await cli.post(
                f"/api/v2/interactions/{request_id}/respond?token=tok",
                json={"choice": "session"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["ok"] is True
            assert body["request_id"] == request_id
            assert body["choice"] == "session"
            mock_resolve.assert_called_once_with("agent:main:3ds:dm:3ds:old3ds:main", "session")
