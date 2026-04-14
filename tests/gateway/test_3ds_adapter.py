import pytest
from unittest.mock import patch
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


def _create_app(adapter):
    app = web.Application()
    app.router.add_get("/api/v2/health", adapter._handle_v2_health)
    app.router.add_get("/api/v2/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/v2/conversations", adapter._handle_conversations)
    app.router.add_post("/api/v2/messages", adapter._handle_messages)
    app.router.add_post("/api/v2/image", adapter._handle_image)
    app.router.add_post("/api/v2/voice", adapter._handle_voice)
    app.router.add_get("/api/v2/events", adapter._handle_events)
    app.router.add_get("/api/v2/media/{media_id}", adapter._handle_media)
    app.router.add_post(
        "/api/v2/interactions/{request_id}/respond",
        adapter._handle_interaction_response,
    )
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


def test_gateway_cli_setup_registry_and_env_example_include_3ds():
    from hermes_cli.gateway import _PLATFORMS

    threeds_platform = next((p for p in _PLATFORMS if p["key"] == "3ds"), None)
    assert threeds_platform is not None
    assert threeds_platform["token_var"] == "THREEDS_ENABLED"

    var_names = [v["name"] for v in threeds_platform["vars"]]
    for key in [
        "THREEDS_ENABLED",
        "THREEDS_HOST",
        "THREEDS_PORT",
        "THREEDS_AUTH_TOKEN",
        "THREEDS_DEVICE_ID",
    ]:
        assert key in var_names

    env_example = (Path(__file__).resolve().parents[2] / ".env.example").read_text()
    assert "THREEDS_ENABLED" in env_example
    assert "THREEDS_HOST" in env_example
    assert "THREEDS_PORT" in env_example
    assert "THREEDS_AUTH_TOKEN" in env_example


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


def test_gateway_runner_disables_streaming_output_for_3ds():
    runner = GatewayRunner(GatewayConfig())

    assert runner._platform_supports_streaming_output(Platform.THREEDS) is False
    assert runner._platform_supports_streaming_output(Platform.DISCORD) is True


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
        assert "model_name" in body
        assert "context_length" in body
        assert "context_tokens" in body
        assert "context_percent" in body

        resp = await cli.get("/api/v2/capabilities?token=tok")
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True
        assert body["transport"] == "http-long-poll"
        assert "model_name" in body
        assert "context_length" in body
        assert "context_tokens" in body
        assert "context_percent" in body


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
@patch("gateway.platforms.threeds.get_model_context_length", return_value=128000)
async def test_health_endpoint_returns_live_session_model_and_context_telemetry(
    _mock_ctx_len, tmp_path
):
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "***"}))
    runner = GatewayRunner(GatewayConfig())
    session_key = adapter._session_key("old3ds", "main")

    class _FakeContextCompressor:
        context_length = 128000
        last_prompt_tokens = 32000

    class _FakeAgent:
        model = "gpt-5.4"
        context_compressor = _FakeContextCompressor()

    runner._running_agents[session_key] = _FakeAgent()
    adapter.gateway_runner = runner
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(
            "/api/v2/health?token=***&device_id=old3ds&conversation_id=main"
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["model_name"] == "gpt-5.4"
        assert body["context_length"] == 128000
        assert body["context_tokens"] == 32000
        assert body["context_percent"] == 25


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
@patch("gateway.platforms.threeds.get_model_context_length", return_value=200000)
async def test_health_endpoint_uses_session_override_and_stored_prompt_tokens_when_no_live_agent(
    _mock_ctx_len, tmp_path
):
    from datetime import datetime

    from gateway.platforms.threeds import ThreeDSAdapter
    from gateway.session import SessionEntry, SessionSource, SessionStore

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "***"}))
    runner = GatewayRunner(GatewayConfig())
    runner.session_store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner.session_store._loaded = True

    session_key = adapter._session_key("old3ds", "focus")
    now = datetime.now()
    runner.session_store._entries = {
        session_key: SessionEntry(
            session_key=session_key,
            session_id="sid-focus",
            created_at=now,
            updated_at=now,
            origin=SessionSource(
                platform=Platform.THREEDS,
                chat_id="3ds:old3ds",
                chat_name="3DS old3ds",
                chat_type="dm",
                user_id="old3ds",
                user_name="old3ds",
                thread_id="focus",
            ),
            display_name="3DS old3ds",
            platform=Platform.THREEDS,
            chat_type="dm",
            last_prompt_tokens=50000,
        )
    }
    runner._session_model_overrides[session_key] = {
        "model": "gpt-5.4-mini",
        "provider": "openai-codex",
        "api_key": "test-key",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }
    adapter.gateway_runner = runner
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(
            "/api/v2/health?token=***&device_id=old3ds&conversation_id=focus"
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["model_name"] == "gpt-5.4-mini"
        assert body["context_length"] == 200000
        assert body["context_tokens"] == 50000
        assert body["context_percent"] == 25


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
@patch("gateway.platforms.threeds.get_model_context_length", return_value=128000)
async def test_health_endpoint_returns_empty_telemetry_for_unknown_session(
    _mock_ctx_len,
):
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(
        PlatformConfig(enabled=True, extra={"auth_token": "tok", "device_id": "old3ds"})
    )
    runner = GatewayRunner(GatewayConfig())
    adapter.gateway_runner = runner
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(
            "/api/v2/health?token=tok&device_id=ghost3ds&conversation_id=missing"
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["model_name"] == ""
        assert body["context_length"] == 0
        assert body["context_tokens"] == 0
        assert body["context_percent"] == 0


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
@patch("gateway.platforms.threeds.get_model_context_length", return_value=128000)
async def test_capabilities_endpoint_returns_session_scoped_telemetry(_mock_ctx_len):
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "tok"}))
    runner = GatewayRunner(GatewayConfig())
    session_key = adapter._session_key("old3ds", "main")

    class _FakeContextCompressor:
        context_length = 128000
        last_prompt_tokens = 6400

    class _FakeAgent:
        model = "gpt-5.4"
        context_compressor = _FakeContextCompressor()

    runner._running_agents[session_key] = _FakeAgent()
    adapter.gateway_runner = runner
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(
            "/api/v2/capabilities?token=tok&device_id=old3ds&conversation_id=main"
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["model_name"] == "gpt-5.4"
        assert body["context_length"] == 128000
        assert body["context_tokens"] == 6400
        assert body["context_percent"] == 5


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_health_endpoint_requires_auth_for_session_scoped_telemetry_when_token_is_configured():
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "tok"}))
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/v2/health?device_id=old3ds&conversation_id=main")
        assert resp.status == 401


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
async def test_partial_updates_emit_message_updated_before_final_created():
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
        ack = await resp.json()

        partial = await adapter.send_partial(
            chat_id="3ds:old3ds",
            content="Hi from",
            reply_to=ack["message_id"],
            metadata={"thread_id": "main"},
        )
        final = await adapter.send(
            chat_id="3ds:old3ds",
            content="Hi from Hermes",
            reply_to=ack["message_id"],
            metadata={"thread_id": "main"},
        )

        assert partial.success is True
        assert final.success is True
        assert partial.message_id == final.message_id

        resp = await cli.get(
            f"/api/v2/events?token=tok&device_id=old3ds&conversation_id=main&cursor={ack['cursor']}&wait=1"
        )
        partial_body = await resp.json()
        assert partial_body["event"]["type"] == "message.updated"
        assert partial_body["event"]["text"] == "Hi from"
        assert partial_body["event"]["reply_to"] == ack["message_id"]

        resp = await cli.get(
            f"/api/v2/events?token=tok&device_id=old3ds&conversation_id=main&cursor={partial_body['cursor']}&wait=1"
        )
        final_body = await resp.json()
        assert final_body["event"]["type"] == "message.created"
        assert final_body["event"]["text"] == "Hi from Hermes"
        assert final_body["event"]["reply_to"] == ack["message_id"]
        assert final_body["event"]["message_id"] == partial.message_id


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_status_updates_emit_deduped_status_updated_events():
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "tok"}))
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await adapter.send_status_update(
            chat_id="3ds:old3ds",
            text="(^_^) pondering...",
            phase="thinking",
            metadata={"thread_id": "main"},
        )
        assert resp.success is True

        duplicate = await adapter.send_status_update(
            chat_id="3ds:old3ds",
            text="(^_^) pondering...",
            phase="thinking",
            metadata={"thread_id": "main"},
        )
        assert duplicate.success is True

        poll = await cli.get(
            "/api/v2/events?token=tok&device_id=old3ds&conversation_id=main&cursor=0&wait=1"
        )
        body = await poll.json()
        assert body["event"]["type"] == "status.updated"
        assert body["event"]["phase"] == "thinking"
        assert body["event"]["text"] == "(^_^) pondering..."

        empty = await cli.get(
            f"/api/v2/events?token=tok&device_id=old3ds&conversation_id=main&cursor={body['cursor']}&wait=1"
        )
        empty_body = await empty.json()
        assert empty_body.get("event") in (None, {})


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_large_message_events_keep_reply_to_ahead_of_text_for_small_clients():
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
                "text": "/help",
            },
        )
        ack = await resp.json()

        large_text = "A" * 7000
        await adapter.send(
            chat_id="3ds:old3ds",
            content=large_text,
            reply_to=ack["message_id"],
            metadata={"thread_id": "main"},
        )

        poll = await cli.get(
            f"/api/v2/events?token=tok&device_id=old3ds&conversation_id=main&cursor={ack['cursor']}&wait=1"
        )
        body = await poll.json()
        event = body["event"]
        assert event["type"] == "message.created"
        keys = list(event.keys())
        assert keys.index("reply_to") < keys.index("text")
        assert event["reply_to"] == ack["message_id"]


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_conversations_endpoint_lists_current_device_conversations_only(tmp_path):
    from datetime import datetime, timedelta

    from hermes_state import SessionDB
    from gateway.platforms.threeds import ThreeDSAdapter
    from gateway.session import SessionEntry, SessionSource, SessionStore

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "***"}))
    session_store = SessionStore(tmp_path / "sessions", GatewayConfig())
    session_store._db = SessionDB(db_path=tmp_path / "state.db")
    session_store._loaded = True

    now = datetime.now()
    session_store._entries = {
        "agent:main:3ds:dm:3ds:old3ds:main": SessionEntry(
            session_key="agent:main:3ds:dm:3ds:old3ds:main",
            session_id="sid-main",
            created_at=now - timedelta(minutes=5),
            updated_at=now - timedelta(minutes=1),
            origin=SessionSource(
                platform=Platform.THREEDS,
                chat_id="3ds:old3ds",
                chat_name="3DS old3ds",
                chat_type="dm",
                user_id="old3ds",
                user_name="old3ds",
                thread_id="main",
            ),
            display_name="3DS old3ds",
            platform=Platform.THREEDS,
            chat_type="dm",
        ),
        "agent:main:3ds:dm:3ds:old3ds:focus": SessionEntry(
            session_key="agent:main:3ds:dm:3ds:old3ds:focus",
            session_id="sid-focus",
            created_at=now - timedelta(minutes=20),
            updated_at=now,
            origin=SessionSource(
                platform=Platform.THREEDS,
                chat_id="3ds:old3ds",
                chat_name="3DS old3ds",
                chat_type="dm",
                user_id="old3ds",
                user_name="old3ds",
                thread_id="focus",
            ),
            display_name="3DS old3ds",
            platform=Platform.THREEDS,
            chat_type="dm",
        ),
        "agent:main:3ds:dm:3ds:other3ds:main": SessionEntry(
            session_key="agent:main:3ds:dm:3ds:other3ds:main",
            session_id="sid-other",
            created_at=now - timedelta(minutes=10),
            updated_at=now - timedelta(minutes=2),
            origin=SessionSource(
                platform=Platform.THREEDS,
                chat_id="3ds:other3ds",
                chat_name="3DS other3ds",
                chat_type="dm",
                user_id="other3ds",
                user_name="other3ds",
                thread_id="main",
            ),
            display_name="3DS other3ds",
            platform=Platform.THREEDS,
            chat_type="dm",
        ),
    }

    session_store._db.create_session("sid-main", "3ds", user_id="old3ds")
    session_store._db.set_session_title("sid-main", "Main Chat")
    session_store._db.append_message("sid-main", "user", "Main conversation prompt")
    session_store._db.create_session("sid-focus", "3ds", user_id="old3ds")
    session_store._db.append_message("sid-focus", "user", "Focus mode question")
    session_store._db.create_session("sid-other", "3ds", user_id="other3ds")
    session_store._db.set_session_title("sid-other", "Other Device")
    session_store._db.append_message("sid-other", "user", "Should not leak")

    adapter.gateway_runner = type(
        "Runner", (), {"session_store": session_store, "_session_db": session_store._db}
    )()
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/v2/conversations?token=***&device_id=old3ds")
        assert resp.status == 200
        body = await resp.json()

    assert body["ok"] is True
    assert body["count"] == 2
    assert [item["conversation_id"] for item in body["conversations"]] == [
        "focus",
        "main",
    ]
    assert body["conversations"][0]["preview"] == "Focus mode question"
    assert body["conversations"][1]["title"] == "Main Chat"
    assert all(item["session_id"] != "sid-other" for item in body["conversations"])


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_voice_upload_creates_voice_message_event_with_audio_attachment():
    from gateway.platforms.threeds import ThreeDSAdapter
    from gateway.platforms.base import MessageType

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "***"}))
    app = _create_app(adapter)
    captured = {}

    async def fake_handle_message(event):
        captured["event"] = event

    adapter.handle_message = fake_handle_message

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/v2/voice?device_id=old3ds&conversation_id=main",
            data=b"RIFFdemoWAVEfmt ",
            headers={
                "Authorization": "Bearer ***",
                "Content-Type": "audio/wav",
            },
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True
        assert body["chat_id"] == "3ds:old3ds"
        assert body["conversation_id"] == "main"
        assert "message_id" in body
        assert "cursor" in body

    event = captured["event"]
    assert event.message_type == MessageType.VOICE
    assert event.media_urls
    assert event.media_types == ["audio/wav"]
    assert event.source.chat_id == "3ds:old3ds"


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_voice_upload_keeps_audio_file_available_after_http_ack():
    from pathlib import Path
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "***"}))
    app = _create_app(adapter)
    captured = {}

    async def fake_handle_message(event):
        captured["path"] = event.media_urls[0]

    adapter.handle_message = fake_handle_message

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/v2/voice?device_id=old3ds&conversation_id=main",
            data=b"RIFFdemoWAVEfmt ",
            headers={
                "Authorization": "Bearer ***",
                "Content-Type": "audio/wav",
            },
        )
        assert resp.status == 200

    uploaded_path = Path(captured["path"])
    assert uploaded_path.exists(), (
        "voice upload file should remain available after request returns"
    )


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_image_upload_creates_photo_message_event_with_image_attachment():
    from gateway.platforms.threeds import ThreeDSAdapter
    from gateway.platforms.base import MessageType

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "***"}))
    app = _create_app(adapter)
    captured = {}

    async def fake_handle_message(event):
        captured["event"] = event

    adapter.handle_message = fake_handle_message

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/v2/image?device_id=old3ds&conversation_id=main",
            data=(
                b"BM>\x00\x00\x00\x00\x00\x00\x006\x00\x00\x00(\x00\x00\x00"
                b"\x02\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00"
                b"\x08\x00\x00\x00\x13\x0b\x00\x00\x13\x0b\x00\x00\x00\x00\x00\x00"
                b"\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00"
            ),
            headers={
                "Authorization": "Bearer ***",
                "Content-Type": "image/bmp",
            },
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["ok"] is True
        assert body["chat_id"] == "3ds:old3ds"
        assert body["conversation_id"] == "main"

    event = captured["event"]
    assert event.message_type == MessageType.PHOTO
    assert event.media_urls
    assert event.media_types == ["image/bmp"]
    assert event.source.chat_id == "3ds:old3ds"


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_send_image_file_emits_media_event_and_serves_preview(tmp_path):
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "tok"}))
    image_path = tmp_path / "sample.bmp"
    image_path.write_bytes(
        b"BM>\x00\x00\x00\x00\x00\x00\x006\x00\x00\x00(\x00\x00\x00"
        b"\x02\x00\x00\x00\x01\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00"
        b"\x08\x00\x00\x00\x13\x0b\x00\x00\x13\x0b\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00"
    )
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        result = await adapter.send_image_file(
            chat_id="3ds:old3ds",
            image_path=str(image_path),
            caption="Look at this",
            reply_to="user-123",
            metadata={"thread_id": "main"},
        )
        assert result.success is True

        poll = await cli.get(
            "/api/v2/events?token=tok&device_id=old3ds&conversation_id=main&cursor=0&wait=1"
        )
        body = await poll.json()
        event = body["event"]
        assert event["type"] == "message.created"
        assert event["reply_to"] == "user-123"
        assert event["text"] == "Look at this"
        assert event["media_id"].startswith("media-")
        assert event["media_type"] == "image/bmp"
        assert event["media_width"] == 2
        assert event["media_height"] == 1

        media_resp = await cli.get(f"/api/v2/media/{event['media_id']}?token=tok")
        assert media_resp.status == 200
        assert media_resp.headers["Content-Type"].startswith("image/bmp")
        assert (await media_resp.read()).startswith(b"BM")


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
        with patch(
            "gateway.platforms.threeds.resolve_gateway_approval", return_value=1
        ) as mock_resolve:
            resp = await cli.post(
                f"/api/v2/interactions/{request_id}/respond?token=tok",
                json={"choice": "session"},
            )
            assert resp.status == 200
            body = await resp.json()
            assert body["ok"] is True
            assert body["request_id"] == request_id
            assert body["choice"] == "session"
            mock_resolve.assert_called_once_with(
                "agent:main:3ds:dm:3ds:old3ds:main", "session"
            )


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_interaction_picker_emits_request_event_and_invokes_callback():
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "tok"}))
    captured: list[str] = []

    async def on_choice(choice: str) -> None:
        captured.append(choice)

    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        result = await adapter.send_interaction_picker(
            chat_id="3ds:old3ds",
            title="Reasoning",
            text="Choose a reasoning level.",
            options=[
                {"choice": f"choice-{index}", "label": f"Option {index}"}
                for index in range(30)
            ],
            on_choice=on_choice,
            metadata={"thread_id": "main"},
            reply_to="user-123",
        )
        assert result.success is True

        poll = await cli.get(
            "/api/v2/events?token=tok&device_id=old3ds&conversation_id=main&cursor=0&wait=1"
        )
        body = await poll.json()
        event = body["event"]
        assert event["type"] == "interaction.request"
        assert event["title"] == "Reasoning"
        assert event["text"] == "Choose a reasoning level."
        assert event["option_count"] == 24
        assert event["choice_0"] == "choice-0"
        assert event["label_23"] == "Option 23"

        request_id = event["request_id"]
        resp = await cli.post(
            f"/api/v2/interactions/{request_id}/respond?token=tok",
            json={"choice": "choice-0"},
        )
        assert resp.status == 200
        response_body = await resp.json()
        assert response_body["ok"] is True
        assert captured == ["choice-0"]


@pytest.mark.asyncio
@patch("gateway.platforms.threeds.AIOHTTP_AVAILABLE", True)
async def test_model_picker_emits_followup_picker_and_final_reply():
    from gateway.platforms.threeds import ThreeDSAdapter

    adapter = ThreeDSAdapter(PlatformConfig(enabled=True, extra={"auth_token": "tok"}))
    chosen: list[tuple[str, str, str]] = []
    providers = [
        {
            "slug": "openrouter",
            "name": "OpenRouter",
            "models": ["openai/gpt-5.4", "anthropic/claude-sonnet-4"],
            "is_current": True,
        }
    ]

    async def on_model_selected(chat_id: str, model_id: str, provider_slug: str) -> str:
        chosen.append((chat_id, model_id, provider_slug))
        return f"Switched to {model_id} via {provider_slug}"

    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        result = await adapter.send_model_picker(
            chat_id="3ds:old3ds",
            providers=providers,
            current_model="gpt-5.4",
            current_provider="openrouter",
            session_key="agent:main:3ds:dm:3ds:old3ds:main",
            on_model_selected=on_model_selected,
            metadata={"thread_id": "main"},
            reply_to="user-999",
        )
        assert result.success is True

        poll1 = await cli.get(
            "/api/v2/events?token=tok&device_id=old3ds&conversation_id=main&cursor=0&wait=1"
        )
        body1 = await poll1.json()
        provider_event = body1["event"]
        assert provider_event["type"] == "interaction.request"
        assert provider_event["choice_0"] == "provider:0"

        resp1 = await cli.post(
            f"/api/v2/interactions/{provider_event['request_id']}/respond?token=tok",
            json={"choice": "provider:0"},
        )
        assert resp1.status == 200

        poll2 = await cli.get(
            f"/api/v2/events?token=tok&device_id=old3ds&conversation_id=main&cursor={body1['cursor']}&wait=1"
        )
        body2 = await poll2.json()
        model_event = body2["event"]
        assert model_event["type"] == "interaction.request"
        assert model_event["choice_0"] == "model:0:0"

        resp2 = await cli.post(
            f"/api/v2/interactions/{model_event['request_id']}/respond?token=tok",
            json={"choice": "model:0:1"},
        )
        assert resp2.status == 200

        poll3 = await cli.get(
            f"/api/v2/events?token=tok&device_id=old3ds&conversation_id=main&cursor={body2['cursor']}&wait=1"
        )
        body3 = await poll3.json()
        final_event = body3["event"]
        assert final_event["type"] == "message.created"
        assert final_event["reply_to"] == "user-999"
        assert "anthropic/claude-sonnet-4" in final_event["text"]
        assert chosen == [("3ds:old3ds", "anthropic/claude-sonnet-4", "openrouter")]
