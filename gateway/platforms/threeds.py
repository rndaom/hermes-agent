"""Native Nintendo 3DS gateway platform adapter.

Exposes a small aiohttp server for the Hermes Agent 3DS client:
- GET  /api/v2/health
- GET  /api/v2/capabilities
- POST /api/v2/messages
- POST /api/v2/voice
- GET  /api/v2/events
- POST /api/v2/interactions/{request_id}/respond
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket as _socket
import uuid
from typing import Any, Dict, Optional

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_audio_from_bytes,
    is_network_accessible,
)
from gateway.session import SessionSource, build_session_key
from tools.approval import resolve_gateway_approval

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
EVENT_QUEUE_LIMIT = 1000
SERVICE_NAME = "hermes-3ds-gateway"
TRANSPORT_NAME = "http-long-poll"


def check_threeds_requirements() -> bool:
    """Check if the 3DS adapter can run."""
    return AIOHTTP_AVAILABLE


class ThreeDSAdapter(BasePlatformAdapter):
    """Hermes gateway adapter for the native 3DS handheld client."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.THREEDS)
        extra = config.extra or {}
        self._host: str = str(extra.get("host", os.getenv("THREEDS_HOST", DEFAULT_HOST)))
        self._port: int = int(extra.get("port", os.getenv("THREEDS_PORT", str(DEFAULT_PORT))))
        self._auth_token: str = str(extra.get("auth_token", os.getenv("THREEDS_AUTH_TOKEN", "")))
        self._default_device_id: str = str(extra.get("device_id", os.getenv("THREEDS_DEVICE_ID", "")))
        self._runner: Optional[web.AppRunner] = None if AIOHTTP_AVAILABLE else None
        self._site = None
        self._app: Optional[web.Application] = None if AIOHTTP_AVAILABLE else None
        self._events: list[dict[str, Any]] = []
        self._cursor: int = 0
        self._event_condition = asyncio.Condition()
        self._pending_interactions: dict[str, dict[str, Any]] = {}

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            return False

        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                sock.connect(("127.0.0.1", self._port))
            logger.error("[3DS] Port %d already in use. Set platforms.3ds.extra.port in config.yaml", self._port)
            return False
        except (ConnectionRefusedError, OSError):
            pass

        if not self._auth_token and is_network_accessible(self._host):
            logger.warning(
                "[3DS] No auth token configured while binding to a network-accessible host (%s). "
                "Set THREEDS_AUTH_TOKEN or platforms.3ds.extra.auth_token for LAN use.",
                self._host,
            )

        self._app = self._build_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        self._mark_connected()
        logger.info("[3DS] Listening on %s:%d", self._host, self._port)
        return True

    async def disconnect(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._site = None
        self._app = None
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        device_id = self._device_id_from_chat_id(chat_id)
        conversation_id = self._conversation_id_from_metadata(metadata)
        message_id = f"assistant-{uuid.uuid4().hex[:12]}"
        event = {
            "device_id": device_id,
            "conversation_id": conversation_id,
            "session_key": self._session_key(device_id, conversation_id),
            "type": "message.created",
            "message_id": message_id,
            "text": content,
            "reply_to": reply_to or "",
        }
        await self._enqueue_event(event)
        return SendResult(success=True, message_id=message_id)

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        device_id = self._device_id_from_chat_id(chat_id)
        conversation_id = self._conversation_id_from_metadata(metadata)
        request_id = f"approval-{uuid.uuid4().hex[:12]}"
        self._pending_interactions[request_id] = {
            "session_key": session_key,
            "device_id": device_id,
            "conversation_id": conversation_id,
            "chat_id": chat_id,
            "command": command,
            "description": description,
        }
        await self._enqueue_event(
            {
                "device_id": device_id,
                "conversation_id": conversation_id,
                "session_key": session_key,
                "type": "approval.request",
                "request_id": request_id,
                "command": command,
                "description": description,
            }
        )
        return SendResult(success=True, message_id=request_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

    def _build_app(self) -> "web.Application":
        app = web.Application()
        app.router.add_get("/api/v2/health", self._handle_v2_health)
        app.router.add_get("/api/v2/capabilities", self._handle_capabilities)
        app.router.add_post("/api/v2/messages", self._handle_messages)
        app.router.add_post("/api/v2/voice", self._handle_voice)
        app.router.add_get("/api/v2/events", self._handle_events)
        app.router.add_post("/api/v2/interactions/{request_id}/respond", self._handle_interaction_response)
        return app

    def _message_ack_payload(self, source: SessionSource, conversation_id: str, message_id: str, cursor: int) -> dict[str, Any]:
        return {
            "ok": True,
            "chat_id": source.chat_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "cursor": cursor,
        }

    def _health_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "platform": "3ds",
            "service": SERVICE_NAME,
            "version": TRANSPORT_NAME,
        }

    def _extract_token(self, request: "web.Request", payload: Optional[dict[str, Any]] = None) -> str:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        token = request.query.get("token", "")
        if token:
            return token
        if payload and isinstance(payload.get("token"), str):
            return payload.get("token", "").strip()
        return ""

    def _is_authorized(self, request: "web.Request", payload: Optional[dict[str, Any]] = None) -> bool:
        if not self._auth_token:
            return True
        return self._extract_token(request, payload) == self._auth_token

    def _device_id_from_chat_id(self, chat_id: str) -> str:
        if chat_id.startswith("3ds:"):
            return chat_id.split(":", 1)[1]
        return chat_id or self._default_device_id or "unknown-3ds"

    def _conversation_id_from_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        if metadata and metadata.get("thread_id"):
            return str(metadata["thread_id"])
        return "main"

    def _session_source(self, device_id: str, conversation_id: str) -> SessionSource:
        return SessionSource(
            platform=Platform.THREEDS,
            chat_id=f"3ds:{device_id}",
            chat_name=f"3DS {device_id}",
            chat_type="dm",
            user_id=device_id,
            user_name=device_id,
            thread_id=conversation_id,
        )

    def _session_key(self, device_id: str, conversation_id: str) -> str:
        return build_session_key(
            self._session_source(device_id, conversation_id),
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
        )

    async def _enqueue_event(self, event: dict[str, Any]) -> int:
        async with self._event_condition:
            self._cursor += 1
            stored = {"cursor": self._cursor, **event}
            self._events.append(stored)
            if len(self._events) > EVENT_QUEUE_LIMIT:
                self._events = self._events[-EVENT_QUEUE_LIMIT:]
            self._event_condition.notify_all()
            return self._cursor

    def _find_matching_event(
        self,
        *,
        after_cursor: int,
        device_id: str,
        conversation_id: str,
    ) -> tuple[Optional[dict[str, Any]], int]:
        matching = [
            event for event in self._events
            if event["cursor"] > after_cursor
            and event.get("device_id") == device_id
            and event.get("conversation_id") == conversation_id
        ]
        if not matching:
            return None, 0

        first = matching[0]
        earliest_visible = matching[0]["cursor"]
        missed_events = max(0, earliest_visible - after_cursor - 1)
        return first, missed_events

    async def _handle_v2_health(self, request: "web.Request") -> "web.Response":
        return web.json_response(self._health_payload())

    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        if not self._is_authorized(request):
            return web.json_response({"ok": False, "error": "Unauthorized."}, status=401)
        return web.json_response(
            {
                "ok": True,
                "platform": "3ds",
                "service": SERVICE_NAME,
                "version": TRANSPORT_NAME,
                "transport": TRANSPORT_NAME,
            }
        )

    async def _handle_messages(self, request: "web.Request") -> "web.Response":
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON."}, status=400)

        if not self._is_authorized(request, payload):
            return web.json_response({"ok": False, "error": "Unauthorized."}, status=401)

        device_id = str(payload.get("device_id", "")).strip() or self._default_device_id
        conversation_id = str(payload.get("conversation_id", "main")).strip() or "main"
        message = str(payload.get("text") or payload.get("message") or "").strip()

        if not device_id:
            return web.json_response({"ok": False, "error": "device_id is required."}, status=400)
        if not message:
            return web.json_response({"ok": False, "error": "text is required."}, status=400)

        source = self._session_source(device_id, conversation_id)
        message_id = f"user-{uuid.uuid4().hex[:12]}"
        ack_cursor = self._cursor
        event = MessageEvent(
            text=message,
            message_type=MessageType.TEXT,
            source=source,
            message_id=message_id,
        )
        await self.handle_message(event)
        return web.json_response(self._message_ack_payload(source, conversation_id, message_id, ack_cursor))

    async def _handle_voice(self, request: "web.Request") -> "web.Response":
        if not self._is_authorized(request):
            return web.json_response({"ok": False, "error": "Unauthorized."}, status=401)

        device_id = request.query.get("device_id", "").strip() or self._default_device_id
        conversation_id = request.query.get("conversation_id", "main").strip() or "main"
        if not device_id:
            return web.json_response({"ok": False, "error": "device_id is required."}, status=400)

        audio_bytes = await request.read()
        if not audio_bytes:
            return web.json_response({"ok": False, "error": "voice body is required."}, status=400)

        content_type = request.content_type or "audio/wav"
        source = self._session_source(device_id, conversation_id)
        message_id = f"user-{uuid.uuid4().hex[:12]}"
        ack_cursor = self._cursor

        suffix = ".wav" if "wav" in content_type else ".bin"
        cached_path = cache_audio_from_bytes(audio_bytes, ext=suffix)
        event = MessageEvent(
            text="",
            message_type=MessageType.VOICE,
            source=source,
            message_id=message_id,
            media_urls=[cached_path],
            media_types=[content_type],
        )
        await self.handle_message(event)
        return web.json_response(self._message_ack_payload(source, conversation_id, message_id, ack_cursor))

    async def _handle_events(self, request: "web.Request") -> "web.Response":
        if not self._is_authorized(request):
            return web.json_response({"ok": False, "error": "Unauthorized."}, status=401)

        device_id = request.query.get("device_id", "").strip() or self._default_device_id
        conversation_id = request.query.get("conversation_id", "main").strip() or "main"
        after_cursor = int(request.query.get("cursor", "0") or "0")
        wait_ms = int(request.query.get("wait", "0") or "0")

        if not device_id:
            return web.json_response({"ok": False, "error": "device_id is required."}, status=400)

        deadline = asyncio.get_running_loop().time() + max(wait_ms, 0) / 1000.0
        async with self._event_condition:
            while True:
                event, missed_events = self._find_matching_event(
                    after_cursor=after_cursor,
                    device_id=device_id,
                    conversation_id=conversation_id,
                )
                if event is not None:
                    return web.json_response(
                        {
                            "ok": True,
                            "cursor": event["cursor"],
                            "missed_events": missed_events,
                            "event": self._event_payload(event),
                        }
                    )

                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return web.json_response(
                        {
                            "ok": True,
                            "cursor": after_cursor,
                            "missed_events": 0,
                        }
                    )
                try:
                    await asyncio.wait_for(self._event_condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return web.json_response(
                        {
                            "ok": True,
                            "cursor": after_cursor,
                            "missed_events": 0,
                        }
                    )

    def _event_payload(self, event: dict[str, Any]) -> dict[str, Any]:
        payload = {"type": event["type"]}
        if event["type"] in {"message.created", "message.updated"}:
            payload.update(
                {
                    "message_id": event.get("message_id", ""),
                    "text": event.get("text", ""),
                    "reply_to": event.get("reply_to", ""),
                }
            )
        elif event["type"] == "approval.request":
            payload.update(
                {
                    "request_id": event.get("request_id", ""),
                    "command": event.get("command", ""),
                    "description": event.get("description", "dangerous command"),
                }
            )
        elif event["type"] == "approval.resolved":
            payload.update(
                {
                    "request_id": event.get("request_id", ""),
                    "choice": event.get("choice", ""),
                }
            )
        return payload

    async def _handle_interaction_response(self, request: "web.Request") -> "web.Response":
        request_id = request.match_info.get("request_id", "")
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON."}, status=400)

        if not self._is_authorized(request, payload):
            return web.json_response({"ok": False, "error": "Unauthorized."}, status=401)

        choice = str(payload.get("choice", "")).strip().lower()
        if choice not in {"once", "session", "always", "deny"}:
            return web.json_response({"ok": False, "error": "Invalid choice."}, status=400)

        pending = self._pending_interactions.pop(request_id, None)
        if pending is None:
            return web.json_response({"ok": False, "error": "Unknown request_id."}, status=404)

        resolve_gateway_approval(pending["session_key"], choice)
        await self._enqueue_event(
            {
                "device_id": pending["device_id"],
                "conversation_id": pending["conversation_id"],
                "session_key": pending["session_key"],
                "type": "approval.resolved",
                "request_id": request_id,
                "choice": choice,
            }
        )
        return web.json_response({"ok": True, "request_id": request_id, "choice": choice})
