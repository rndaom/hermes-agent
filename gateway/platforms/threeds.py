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
import queue
import socket as _socket
import time
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
from agent.model_metadata import get_model_context_length

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
EVENT_QUEUE_LIMIT = 1000
SERVICE_NAME = "hermes-3ds-gateway"
TRANSPORT_NAME = "http-long-poll"
_STREAM_DONE = object()


class ThreeDSStreamConsumer:
    """Emit throttled `message.updated` events while Hermes is generating."""

    def __init__(
        self,
        adapter: "ThreeDSAdapter",
        chat_id: str,
        reply_to: str,
        *,
        edit_interval: float,
        buffer_threshold: int,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.adapter = adapter
        self.chat_id = chat_id
        self.reply_to = reply_to
        self.edit_interval = edit_interval
        self.buffer_threshold = buffer_threshold
        self.metadata = metadata
        self._queue: queue.Queue = queue.Queue()
        self._accumulated = ""
        self._last_sent_text = ""
        self._last_emit = 0.0
        self._already_sent = False
        self._final_response_sent = False

    @property
    def already_sent(self) -> bool:
        return self._already_sent

    @property
    def final_response_sent(self) -> bool:
        return self._final_response_sent

    def on_delta(self, text: Optional[str]) -> None:
        if text:
            self._queue.put(text)

    def on_segment_break(self) -> None:
        return

    def on_commentary(self, text: str) -> None:
        return

    def finish(self) -> None:
        self._queue.put(_STREAM_DONE)

    async def _flush_partial(self) -> None:
        if not self._accumulated or self._accumulated == self._last_sent_text:
            return
        result = await self.adapter.send_partial(
            chat_id=self.chat_id,
            content=self._accumulated,
            reply_to=self.reply_to,
            metadata=self.metadata,
        )
        if result.success:
            self._already_sent = True
            self._last_sent_text = self._accumulated
            self._last_emit = time.monotonic()

    async def run(self) -> None:
        while True:
            got_done = False
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is _STREAM_DONE:
                    got_done = True
                    break
                self._accumulated += str(item)

            delta_size = len(self._accumulated) - len(self._last_sent_text)
            elapsed = time.monotonic() - self._last_emit
            should_flush = got_done or (
                delta_size >= self.buffer_threshold
                or (delta_size > 0 and elapsed >= self.edit_interval)
            )

            if should_flush:
                await self._flush_partial()
            if got_done:
                return
            await asyncio.sleep(0.05)


def check_threeds_requirements() -> bool:
    """Check if the 3DS adapter can run."""
    return AIOHTTP_AVAILABLE


class ThreeDSAdapter(BasePlatformAdapter):
    """Hermes gateway adapter for the native 3DS handheld client."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.THREEDS)
        extra = config.extra or {}
        self._host: str = str(
            extra.get("host", os.getenv("THREEDS_HOST", DEFAULT_HOST))
        )
        self._port: int = int(
            extra.get("port", os.getenv("THREEDS_PORT", str(DEFAULT_PORT)))
        )
        self._auth_token: str = str(
            extra.get("auth_token", os.getenv("THREEDS_AUTH_TOKEN", ""))
        )
        self._default_device_id: str = str(
            extra.get("device_id", os.getenv("THREEDS_DEVICE_ID", ""))
        )
        self._runner: Optional[web.AppRunner] = None if AIOHTTP_AVAILABLE else None
        self._site = None
        self._app: Optional[web.Application] = None if AIOHTTP_AVAILABLE else None
        self._events: list[dict[str, Any]] = []
        self._cursor: int = 0
        self._event_condition = asyncio.Condition()
        self._pending_interactions: dict[str, dict[str, Any]] = {}
        self._pending_stream_message_ids: dict[str, str] = {}

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            return False

        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                sock.connect(("127.0.0.1", self._port))
            logger.error(
                "[3DS] Port %d already in use. Set platforms.3ds.extra.port in config.yaml",
                self._port,
            )
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
        stream_key = self._stream_key(device_id, conversation_id, reply_to)
        message_id = (
            self._pending_stream_message_ids.pop(stream_key, None)
            or f"assistant-{uuid.uuid4().hex[:12]}"
        )
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

    def _stream_key(
        self, device_id: str, conversation_id: str, reply_to: Optional[str]
    ) -> str:
        return f"{device_id}:{conversation_id}:{reply_to or ''}"

    def _stream_message_id(
        self, device_id: str, conversation_id: str, reply_to: str
    ) -> str:
        stream_key = self._stream_key(device_id, conversation_id, reply_to)
        message_id = self._pending_stream_message_ids.get(stream_key)
        if message_id is None:
            message_id = f"assistant-{uuid.uuid4().hex[:12]}"
            self._pending_stream_message_ids[stream_key] = message_id
            if len(self._pending_stream_message_ids) > EVENT_QUEUE_LIMIT:
                self._pending_stream_message_ids.pop(
                    next(iter(self._pending_stream_message_ids))
                )
        return message_id

    async def send_partial(
        self,
        chat_id: str,
        content: str,
        reply_to: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        device_id = self._device_id_from_chat_id(chat_id)
        conversation_id = self._conversation_id_from_metadata(metadata)

        if not reply_to:
            return SendResult(
                success=False, error="reply_to is required for partial 3DS updates"
            )

        message_id = self._stream_message_id(device_id, conversation_id, reply_to)
        await self._enqueue_event(
            {
                "device_id": device_id,
                "conversation_id": conversation_id,
                "session_key": self._session_key(device_id, conversation_id),
                "type": "message.updated",
                "message_id": message_id,
                "text": content,
                "reply_to": reply_to,
            }
        )
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
        app.router.add_get("/api/v2/conversations", self._handle_conversations)
        app.router.add_post("/api/v2/messages", self._handle_messages)
        app.router.add_post("/api/v2/voice", self._handle_voice)
        app.router.add_get("/api/v2/events", self._handle_events)
        app.router.add_post(
            "/api/v2/interactions/{request_id}/respond",
            self._handle_interaction_response,
        )
        return app

    def _message_ack_payload(
        self, source: SessionSource, conversation_id: str, message_id: str, cursor: int
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "chat_id": source.chat_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "cursor": cursor,
        }

    def _health_payload(
        self, device_id: str = "", conversation_id: str = "main"
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "platform": "3ds",
            "service": SERVICE_NAME,
            "version": TRANSPORT_NAME,
            **self._telemetry_payload(device_id, conversation_id),
        }

    def _session_entry_for(self, device_id: str, conversation_id: str):
        runner = getattr(self, "gateway_runner", None)
        session_store = (
            getattr(runner, "session_store", None) if runner is not None else None
        )
        session_key = self._session_key(device_id, conversation_id)
        if session_store is None:
            return session_key, None
        try:
            for entry in session_store.list_sessions():
                if getattr(entry, "session_key", "") == session_key:
                    return session_key, entry
        except Exception as exc:
            logger.debug("[3DS] Failed to list sessions for %s: %s", session_key, exc)
        entries = getattr(session_store, "_entries", {})
        return session_key, entries.get(session_key)

    def _cached_or_running_agent_for(self, session_key: str):
        runner = getattr(self, "gateway_runner", None)
        if runner is None:
            return None

        agent = getattr(runner, "_running_agents", {}).get(session_key)
        if agent is not None and hasattr(agent, "model"):
            return agent

        cache_lock = getattr(runner, "_agent_cache_lock", None)
        cache = getattr(runner, "_agent_cache", None)
        if cache_lock is None or cache is None:
            return None

        with cache_lock:
            cached = cache.get(session_key)
        if not cached:
            return None
        if isinstance(cached, tuple):
            cached = cached[0]
        if cached is not None and hasattr(cached, "model"):
            return cached
        return None

    def _telemetry_payload(
        self, device_id: str = "", conversation_id: str = "main"
    ) -> dict[str, Any]:
        resolved_device_id = (device_id or "").strip()
        resolved_conversation_id = (conversation_id or "main").strip() or "main"
        model_name = ""
        context_length = 0
        context_tokens = 0
        context_percent = 0
        runtime_kwargs: dict[str, Any] = {}

        if not resolved_device_id:
            return {
                "model_name": model_name,
                "context_length": context_length,
                "context_tokens": context_tokens,
                "context_percent": context_percent,
            }

        runner = getattr(self, "gateway_runner", None)
        session_key, session_entry = self._session_entry_for(
            resolved_device_id, resolved_conversation_id
        )
        agent = self._cached_or_running_agent_for(session_key)
        if agent is None and session_entry is None:
            return {
                "model_name": model_name,
                "context_length": context_length,
                "context_tokens": context_tokens,
                "context_percent": context_percent,
            }

        source = self._session_source(resolved_device_id, resolved_conversation_id)
        if runner is not None and hasattr(runner, "_resolve_session_agent_runtime"):
            try:
                model_name, runtime_kwargs = runner._resolve_session_agent_runtime(
                    source=source,
                    session_key=session_key,
                )
            except Exception as exc:
                logger.debug(
                    "[3DS] Failed to resolve session runtime for %s: %s",
                    session_key,
                    exc,
                )
                model_name = ""
                runtime_kwargs = {}

        if agent is not None:
            model_name = getattr(agent, "model", model_name) or model_name
            context_compressor = getattr(agent, "context_compressor", None)
            if context_compressor is not None:
                context_length = int(
                    getattr(context_compressor, "context_length", 0) or 0
                )
                context_tokens = int(
                    getattr(context_compressor, "last_prompt_tokens", 0) or 0
                )

        if context_tokens == 0 and session_entry is not None:
            context_tokens = int(getattr(session_entry, "last_prompt_tokens", 0) or 0)

        if model_name and context_length <= 0:
            try:
                context_length = int(
                    get_model_context_length(
                        model_name,
                        base_url=(runtime_kwargs.get("base_url") or ""),
                        api_key=(runtime_kwargs.get("api_key") or ""),
                        provider=(runtime_kwargs.get("provider") or ""),
                    )
                    or 0
                )
            except Exception as exc:
                logger.debug(
                    "[3DS] Failed to resolve context length for %s: %s", model_name, exc
                )
                context_length = 0

        if context_length > 0 and context_tokens > 0:
            context_percent = max(
                0, min(100, round((context_tokens / context_length) * 100))
            )
        elif context_length > 0:
            context_percent = 0

        return {
            "model_name": model_name,
            "context_length": context_length,
            "context_tokens": context_tokens,
            "context_percent": context_percent,
        }

    def _extract_token(
        self, request: "web.Request", payload: Optional[dict[str, Any]] = None
    ) -> str:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        token = request.query.get("token", "")
        if token:
            return token
        if payload and isinstance(payload.get("token"), str):
            return payload.get("token", "").strip()
        return ""

    def _is_authorized(
        self, request: "web.Request", payload: Optional[dict[str, Any]] = None
    ) -> bool:
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
            group_sessions_per_user=self.config.extra.get(
                "group_sessions_per_user", True
            ),
            thread_sessions_per_user=self.config.extra.get(
                "thread_sessions_per_user", False
            ),
        )

    @staticmethod
    def _compact_preview(text: str, limit: int = 60) -> str:
        compact = " ".join((text or "").split())
        if len(compact) <= limit:
            return compact
        return compact[:limit].rstrip() + "..."

    def _preview_for_session(self, session_id: str) -> str:
        runner = getattr(self, "gateway_runner", None)
        if runner is None or getattr(runner, "session_store", None) is None:
            return ""

        try:
            history = runner.session_store.load_transcript(session_id) or []
        except Exception as exc:
            logger.debug(
                "[3DS] Failed to load transcript preview for %s: %s", session_id, exc
            )
            return ""

        for index in range(len(history), 0, -1):
            message = history[index - 1]
            if message.get("role") == "user" and message.get("content"):
                return self._compact_preview(str(message.get("content", "")))
        return ""

    def _list_conversations(self, device_id: str, limit: int) -> list[dict[str, Any]]:
        runner = getattr(self, "gateway_runner", None)
        if runner is None or getattr(runner, "session_store", None) is None:
            return []

        session_db = getattr(runner, "_session_db", None)
        expected_chat_id = f"3ds:{device_id}"
        conversations: list[dict[str, Any]] = []

        for entry in runner.session_store.list_sessions():
            if entry.platform != Platform.THREEDS or entry.origin is None:
                continue
            if (
                entry.origin.chat_id != expected_chat_id
                and entry.origin.user_id != device_id
            ):
                continue

            conversation_id = (entry.origin.thread_id or "main").strip() or "main"
            title = ""
            if session_db is not None:
                try:
                    title = session_db.get_session_title(entry.session_id) or ""
                except Exception as exc:
                    logger.debug(
                        "[3DS] Failed to fetch title for %s: %s", entry.session_id, exc
                    )

            conversations.append(
                {
                    "conversation_id": conversation_id,
                    "session_id": entry.session_id,
                    "title": title,
                    "preview": self._preview_for_session(entry.session_id),
                    "updated_at": entry.updated_at.isoformat()
                    if entry.updated_at
                    else None,
                    "_sort_ts": entry.updated_at.timestamp() if entry.updated_at else 0,
                }
            )

        conversations.sort(
            key=lambda item: (
                item.get("_sort_ts") or 0,
                item.get("conversation_id") or "",
            ),
            reverse=True,
        )
        trimmed = conversations[:limit]
        for item in trimmed:
            item.pop("_sort_ts", None)
        return trimmed

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
            event
            for event in self._events
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
        raw_device_id = request.query.get("device_id", "").strip()
        raw_conversation_id = request.query.get("conversation_id", "").strip()
        telemetry_requested = bool(raw_device_id or raw_conversation_id)
        if telemetry_requested and not self._is_authorized(request):
            return web.json_response(
                {"ok": False, "error": "Unauthorized."}, status=401
            )
        if not telemetry_requested:
            return web.json_response(self._health_payload("", "main"))
        device_id = raw_device_id or self._default_device_id
        conversation_id = raw_conversation_id or "main"
        return web.json_response(self._health_payload(device_id, conversation_id))

    async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
        if not self._is_authorized(request):
            return web.json_response(
                {"ok": False, "error": "Unauthorized."}, status=401
            )
        device_id = (
            request.query.get("device_id", "").strip() or self._default_device_id
        )
        conversation_id = request.query.get("conversation_id", "main").strip() or "main"
        return web.json_response(
            {
                "ok": True,
                "platform": "3ds",
                "service": SERVICE_NAME,
                "version": TRANSPORT_NAME,
                "transport": TRANSPORT_NAME,
                **self._telemetry_payload(device_id, conversation_id),
            }
        )

    async def _handle_conversations(self, request: "web.Request") -> "web.Response":
        if not self._is_authorized(request):
            return web.json_response(
                {"ok": False, "error": "Unauthorized."}, status=401
            )

        device_id = (
            request.query.get("device_id", "").strip() or self._default_device_id
        )
        if not device_id:
            return web.json_response(
                {"ok": False, "error": "device_id is required."}, status=400
            )

        try:
            requested_limit = int(request.query.get("limit", "8") or "8")
        except ValueError:
            requested_limit = 8
        limit = max(1, min(requested_limit, 20))
        conversations = self._list_conversations(device_id, limit)
        return web.json_response(
            {
                "ok": True,
                "count": len(conversations),
                "conversations": conversations,
            }
        )

    async def _handle_messages(self, request: "web.Request") -> "web.Response":
        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "Invalid JSON."}, status=400
            )

        if not self._is_authorized(request, payload):
            return web.json_response(
                {"ok": False, "error": "Unauthorized."}, status=401
            )

        device_id = str(payload.get("device_id", "")).strip() or self._default_device_id
        conversation_id = str(payload.get("conversation_id", "main")).strip() or "main"
        message = str(payload.get("text") or payload.get("message") or "").strip()

        if not device_id:
            return web.json_response(
                {"ok": False, "error": "device_id is required."}, status=400
            )
        if not message:
            return web.json_response(
                {"ok": False, "error": "text is required."}, status=400
            )

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
        return web.json_response(
            self._message_ack_payload(source, conversation_id, message_id, ack_cursor)
        )

    async def _handle_voice(self, request: "web.Request") -> "web.Response":
        if not self._is_authorized(request):
            return web.json_response(
                {"ok": False, "error": "Unauthorized."}, status=401
            )

        device_id = (
            request.query.get("device_id", "").strip() or self._default_device_id
        )
        conversation_id = request.query.get("conversation_id", "main").strip() or "main"
        if not device_id:
            return web.json_response(
                {"ok": False, "error": "device_id is required."}, status=400
            )

        audio_bytes = await request.read()
        if not audio_bytes:
            return web.json_response(
                {"ok": False, "error": "voice body is required."}, status=400
            )

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
        return web.json_response(
            self._message_ack_payload(source, conversation_id, message_id, ack_cursor)
        )

    async def _handle_events(self, request: "web.Request") -> "web.Response":
        if not self._is_authorized(request):
            return web.json_response(
                {"ok": False, "error": "Unauthorized."}, status=401
            )

        device_id = (
            request.query.get("device_id", "").strip() or self._default_device_id
        )
        conversation_id = request.query.get("conversation_id", "main").strip() or "main"
        after_cursor = int(request.query.get("cursor", "0") or "0")
        wait_ms = int(request.query.get("wait", "0") or "0")

        if not device_id:
            return web.json_response(
                {"ok": False, "error": "device_id is required."}, status=400
            )

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
                    await asyncio.wait_for(
                        self._event_condition.wait(), timeout=remaining
                    )
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

    async def _handle_interaction_response(
        self, request: "web.Request"
    ) -> "web.Response":
        request_id = request.match_info.get("request_id", "")
        try:
            payload = await request.json()
        except Exception:
            return web.json_response(
                {"ok": False, "error": "Invalid JSON."}, status=400
            )

        if not self._is_authorized(request, payload):
            return web.json_response(
                {"ok": False, "error": "Unauthorized."}, status=401
            )

        choice = str(payload.get("choice", "")).strip().lower()
        if choice not in {"once", "session", "always", "deny"}:
            return web.json_response(
                {"ok": False, "error": "Invalid choice."}, status=400
            )

        pending = self._pending_interactions.pop(request_id, None)
        if pending is None:
            return web.json_response(
                {"ok": False, "error": "Unknown request_id."}, status=404
            )

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
        return web.json_response(
            {"ok": True, "request_id": request_id, "choice": choice}
        )
