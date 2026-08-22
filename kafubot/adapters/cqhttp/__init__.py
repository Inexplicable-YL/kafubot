from __future__ import annotations

import json
import sys
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar
from typing_extensions import override

import aiohttp
import anyio
from aiohttp import web
from pydantic import BaseModel, ConfigDict

from kafubot.adapters.utils import WebSocketAdapter
from kafubot.config import CQHTTPConfig
from kafubot.log import logger

if TYPE_CHECKING:
    from kafubot.bot import Bot

from .event import (
    DEFAULT_EVENT_MODELS,
    CQHTTPEvent,
    EventModels,
    HeartbeatMetaEvent,
    LifecycleMetaEvent,
    MessageEvent,
    Reply,
    parse_event,
)
from .exceptions import (
    ActionFailed,
    ApiNotAvailable,
    ApiTimeout,
    NetworkError,
)
from .message import CQHTTPMessage, CQHTTPMessageSegment

__all__ = ["CQHTTPAdapter"]


class _PendingResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ready: anyio.Event
    response: dict[str, Any] | None = None
    error: Exception | None = None


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, CQHTTPMessage):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class CQHTTPAdapter(WebSocketAdapter[CQHTTPEvent, CQHTTPConfig]):
    """OneBot v11 WebSocket adapter with forward and reverse WS modes."""

    name = "cqhttp"
    Config = CQHTTPConfig
    event_models: ClassVar[EventModels] = dict(DEFAULT_EVENT_MODELS)

    def __init__(self, bot: Bot) -> None:
        super().__init__(bot)
        self._api_id = 0
        self._pending: dict[int, _PendingResponse] = {}
        self._send_lock = anyio.Lock()

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        return partial(self.call_api, item)

    @classmethod
    def add_event_model(cls, event_model: type[CQHTTPEvent]) -> None:
        """Register a custom OneBot event model, matching SekaiBot's API."""
        if not issubclass(event_model, CQHTTPEvent):
            raise TypeError("event_model must inherit CQHTTPEvent")
        cls.event_models[event_model.get_event_type()] = event_model

    @classmethod
    def get_event_model(
        cls,
        post_type: str | None,
        detail_type: str | None,
        sub_type: str | None,
    ) -> type[CQHTTPEvent]:
        """Resolve the most specific registered model for an event type tuple."""
        return (
            cls.event_models.get((post_type, detail_type, sub_type))
            or cls.event_models.get((post_type, detail_type, None))
            or cls.event_models.get((post_type, None, None))
            or cls.event_models.get((None, None, None), CQHTTPEvent)
        )

    @override
    async def startup(self) -> None:
        self.adapter_type = (
            "reverse-ws"
            if self.config.adapter_type == "ws-reverse"
            else self.config.adapter_type
        )
        self.host = self.config.host
        self.port = self.config.port
        self.url = self.config.url
        self.reconnect_interval = self.config.reconnect_interval
        self._pending.clear()
        await super().startup()

    @override
    async def shutdown(self) -> None:
        self._fail_pending(NetworkError("CQHTTP connection closed"))
        await super().shutdown()

    @override
    def retry_delay(self, retries: int) -> float:
        _ = retries
        return self.config.reconnect_interval

    @override
    async def handle_reverse_ws_response(
        self,
        request: web.Request,
    ) -> web.WebSocketResponse:
        if self.config.access_token and request.headers.get("Authorization") != (
            f"Bearer {self.config.access_token}"
        ):
            logger.warning("Rejected CQHTTP WebSocket: access token mismatch")
            raise web.HTTPUnauthorized(text="CQHTTP access token mismatch")
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        if self.websocket is not None and not self.websocket.closed:
            logger.warning("Replacing existing CQHTTP WebSocket connection")
            self._fail_pending(NetworkError("CQHTTP connection replaced"))
            await self.websocket.close()
        self.websocket = websocket
        await self.reverse_ws_connection_hook()
        try:
            await self._consume_websocket(websocket)
        finally:
            if self.websocket is websocket:
                self.websocket = None
                self._fail_pending(NetworkError("CQHTTP connection closed"))
            logger.warning("CQHTTP WebSocket disconnected")
        return websocket

    async def _accept_reverse_ws(self, request: web.Request) -> web.WebSocketResponse:
        """Compatibility alias for the previous local transport implementation."""
        return await self.handle_reverse_ws_response(request)

    @override
    async def reverse_ws_connection_hook(self) -> None:
        """Compatibility hook run after an authenticated reverse WS connects."""
        logger.info("CQHTTP WebSocket connected", mode="reverse-ws")

    @override
    async def websocket_connect(self) -> None:
        """Open and consume one forward WebSocket connection."""
        if self.session is None:
            raise NetworkError("CQHTTP client session is not initialized")
        headers = (
            {"Authorization": f"Bearer {self.config.access_token}"}
            if self.config.access_token
            else None
        )
        url = f"ws://{self.config.host}:{self.config.port}{self.config.url}"
        logger.info("Connecting to CQHTTP", url=url)
        async with self.session.ws_connect(url, headers=headers) as websocket:
            self.websocket = websocket
            logger.info("CQHTTP WebSocket connected", mode="ws")
            await self._consume_websocket(websocket)
        self._fail_pending(NetworkError("CQHTTP connection closed"))

    @override
    async def handle_websocket_msg(self, message: aiohttp.WSMessage) -> None:
        """Handle one WebSocket frame using SekaiBot's former hook name."""
        if message.type == aiohttp.WSMsgType.TEXT:
            await self._handle_text(str(message.data))
            return
        if message.type == aiohttp.WSMsgType.ERROR:
            websocket = self.websocket
            error = websocket.exception() if websocket is not None else None
            raise NetworkError(str(error or "WebSocket receive failed"))

    async def _handle_text(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid CQHTTP JSON")
            return
        if not isinstance(payload, dict):
            logger.warning("Ignoring non-object CQHTTP payload")
            return
        if "post_type" not in payload:
            echo = payload.get("echo")
            if isinstance(echo, int) and (pending := self._pending.get(echo)):
                pending.response = payload
                pending.ready.set()
            return
        await self.handle_cqhttp_event(payload)

    async def handle_cqhttp_event(self, payload: dict[str, Any]) -> None:
        try:
            event = parse_event(self, payload, self.event_models)
        except Exception:
            logger.exception("Failed to parse CQHTTP event", payload=payload)
            return

        if isinstance(event, LifecycleMetaEvent):
            if event.sub_type == "connect":
                logger.info("CQHTTP lifecycle connected", self_id=event.self_id)
            return
        if isinstance(event, HeartbeatMetaEvent):
            if not (event.status.good and event.status.online):
                logger.error(
                    "CQHTTP heartbeat unhealthy", status=event.status.model_dump()
                )
            return
        if isinstance(event, MessageEvent):
            await self._get_reply(event)
            await self._get_at_me(event)
        await self.handle_event(event)

    async def _get_reply(self, event: MessageEvent) -> None:
        """SekaiBot-compatible name for reply extraction."""
        await self._extract_reply(event)

    async def _extract_reply(self, event: MessageEvent) -> None:
        index = next(
            (
                index
                for index, segment in enumerate(event.message)
                if segment.type == "reply"
            ),
            None,
        )
        if index is None:
            return
        segment = event.message[index]
        try:
            response = await self.call_api(
                "get_msg", message_id=int(segment.data["id"])
            )
            event.reply = Reply.model_validate(response)
        except Exception as exc:
            logger.warning("Could not resolve quoted message", error=repr(exc))
            return

        replied_user = event.reply.sender.user_id
        if replied_user is not None:
            if str(replied_user) == str(event.self_id):
                event.to_me = True
            del event.message[index]
            if (
                len(event.message) > index
                and event.message[index].type == "at"
                and event.message[index].data.get("qq") == str(replied_user)
            ):
                del event.message[index]
        self._trim_leading_text(event, index)

    @classmethod
    def _trim_leading_text(cls, event: MessageEvent, index: int) -> None:
        if len(event.message) > index and event.message[index].type == "text":
            text = str(event.message[index].data.get("text", "")).lstrip()
            if text:
                event.message[index].data["text"] = text
            else:
                del event.message[index]
        if not event.message:
            event.message.append(CQHTTPMessageSegment.text(""))

    @classmethod
    def _extract_at_me(cls, event: MessageEvent) -> None:
        if not event.message:
            event.message.append(CQHTTPMessageSegment.text(""))
        if event.message_type == "private":
            event.to_me = True
            return

        def is_at_me(segment: CQHTTPMessageSegment) -> bool:
            return segment.type == "at" and str(segment.data.get("qq", "")) == str(
                event.self_id
            )

        while event.message and is_at_me(event.message[0]):
            event.to_me = True
            event.message.pop(0)
            if event.message and event.message[0].type == "text":
                text = str(event.message[0].data.get("text", "")).lstrip()
                if text:
                    event.message[0].data["text"] = text
                else:
                    event.message.pop(0)

        if not event.to_me and event.message:
            index = -1
            if (
                event.message[index].type == "text"
                and not str(event.message[index].data.get("text", "")).strip()
                and len(event.message) >= 2
            ):
                index = -2
            if is_at_me(event.message[index]):
                event.to_me = True
                del event.message[index:]
        if not event.message:
            event.message.append(CQHTTPMessageSegment.text(""))

    async def _get_at_me(self, event: MessageEvent) -> None:
        """SekaiBot-compatible name for at-mention extraction."""
        self._extract_at_me(event)

    def _next_echo(self) -> int:
        self._api_id = (self._api_id + 1) % sys.maxsize
        return self._api_id

    def _get_api_echo(self) -> int:
        """SekaiBot-compatible name for allocating an API echo."""
        return self._next_echo()

    @override
    async def _call_api(self, api: str, **params: Any) -> Any:
        websocket = self.websocket
        if websocket is None or websocket.closed:
            raise NetworkError("CQHTTP WebSocket is not connected")
        echo = self._get_api_echo()
        pending = _PendingResponse(ready=anyio.Event())
        self._pending[echo] = pending
        request = json.dumps(
            {"action": api, "params": params, "echo": echo},
            ensure_ascii=False,
            default=_json_default,
        )
        try:
            async with self._send_lock:
                await websocket.send_str(request)
            with anyio.fail_after(self.config.api_timeout):
                await pending.ready.wait()
        except TimeoutError as exc:
            raise ApiTimeout(f"OneBot API timed out: {api}") from exc
        except ApiTimeout:
            raise
        except Exception as exc:
            raise NetworkError(f"OneBot API transport failed: {api}") from exc
        finally:
            self._pending.pop(echo, None)

        if pending.error is not None:
            raise pending.error
        response = pending.response or {}
        if response.get("retcode") == ApiNotAvailable.ERROR_CODE:
            raise ApiNotAvailable(response)
        if response.get("status") == "failed":
            raise ActionFailed(response)
        return response.get("data")

    def _fail_pending(self, error: Exception) -> None:
        for pending in self._pending.values():
            pending.error = error
            pending.ready.set()

    @override
    async def send(
        self,
        event: Any,
        message: Any,
        at_sender: bool = False,
        reply_message: bool = False,
        **params: Any,
    ) -> Any:
        if not isinstance(event, CQHTTPEvent):
            raise TypeError(f"CQHTTP cannot send for {type(event)!r}")
        event_data = event.model_dump()
        if "message_id" not in event_data:
            reply_message = False
        if "user_id" in event_data:
            params.setdefault("user_id", event_data["user_id"])
        else:
            at_sender = False
        if "group_id" in event_data:
            params.setdefault("group_id", event_data["group_id"])
        if "message_type" in event_data:
            params.setdefault("message_type", event_data["message_type"])
        if "message_type" not in params:
            if params.get("group_id") is not None:
                params["message_type"] = "group"
            elif params.get("user_id") is not None:
                params["message_type"] = "private"
            else:
                raise ValueError("cannot infer OneBot message type")

        full_message = CQHTTPMessage()
        if reply_message:
            full_message += CQHTTPMessageSegment.reply(event_data["message_id"])
        if at_sender and params["message_type"] != "private":
            full_message += CQHTTPMessageSegment.at(params["user_id"]) + " "
        full_message += message
        params.setdefault("message", full_message)
        return await self.call_api("send_msg", **params)
