from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Literal
from typing_extensions import override

import aiohttp
import anyio
from aiohttp import web
from anyio.lowlevel import checkpoint

from kafubot.log import logger

from . import Adapter, ConfigT, EventT

if TYPE_CHECKING:
    from anyio.streams.memory import (
        MemoryObjectReceiveStream,
        MemoryObjectSendStream,
    )

__all__ = [
    "HttpClientAdapter",
    "HttpServerAdapter",
    "PollingAdapter",
    "WebSocketAdapter",
    "WebSocketClientAdapter",
    "WebSocketServerAdapter",
]


class PollingAdapter(Adapter[EventT, ConfigT], metaclass=ABCMeta):
    """Base for adapters that poll an external protocol."""

    @override
    async def run(self) -> None:
        while not self.should_stop:
            await checkpoint()
            await self.on_tick()

    @abstractmethod
    async def on_tick(self) -> None:
        """Run one polling iteration."""


class HttpClientAdapter(PollingAdapter[EventT, ConfigT], metaclass=ABCMeta):
    """Polling adapter with a managed aiohttp client session."""

    session: aiohttp.ClientSession

    @override
    async def startup(self) -> None:
        self.session = aiohttp.ClientSession()

    @override
    async def shutdown(self) -> None:
        if not self.session.closed:
            await self.session.close()


class WebSocketClientAdapter(Adapter[EventT, ConfigT], metaclass=ABCMeta):
    """Simple WebSocket client adapter."""

    url: str

    @override
    async def run(self) -> None:
        async with (
            aiohttp.ClientSession() as session,
            session.ws_connect(self.url) as websocket,
        ):
            async for message in websocket:
                await checkpoint()
                if message.type == aiohttp.WSMsgType.ERROR:
                    break
                await self.handle_response(message)

    @abstractmethod
    async def handle_response(self, message: aiohttp.WSMessage) -> None:
        """Handle one WebSocket response."""


class HttpServerAdapter(Adapter[EventT, ConfigT], metaclass=ABCMeta):
    """HTTP server adapter with GET and POST entry points."""

    app: web.Application
    runner: web.AppRunner
    site: web.TCPSite
    host: str
    port: int
    get_url: str
    post_url: str

    @override
    async def startup(self) -> None:
        self.app = web.Application()
        self.app.add_routes(
            [
                web.get(self.get_url, self.handle_response),
                web.post(self.post_url, self.handle_response),
            ]
        )

    @override
    async def run(self) -> None:
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        await self.wait_stopped()

    @override
    async def shutdown(self) -> None:
        if hasattr(self, "runner"):
            await self.runner.cleanup()

    @abstractmethod
    async def handle_response(self, request: web.Request) -> web.StreamResponse:
        """Handle one HTTP request."""


class WebSocketServerAdapter(Adapter[EventT, ConfigT], metaclass=ABCMeta):
    """WebSocket server adapter with concurrent frame dispatch."""

    app: web.Application
    runner: web.AppRunner
    site: web.TCPSite
    websocket: web.WebSocketResponse | None = None
    host: str
    port: int
    url: str

    _msg_send_stream: MemoryObjectSendStream[aiohttp.WSMessage]
    _msg_receive_stream: MemoryObjectReceiveStream[aiohttp.WSMessage]

    @override
    async def startup(self) -> None:
        self.app = web.Application()
        self.app.add_routes([web.get(self.url, self.handle_response)])
        self._msg_send_stream, self._msg_receive_stream = (
            anyio.create_memory_object_stream[aiohttp.WSMessage](
                max_buffer_size=float("inf")
            )
        )

    @override
    async def run(self) -> None:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(self._handle_msg_receive)
            try:
                self.runner = web.AppRunner(self.app)
                await self.runner.setup()
                self.site = web.TCPSite(self.runner, self.host, self.port)
                await self.site.start()
                await self.wait_stopped()
            finally:
                await self._msg_send_stream.aclose()

    @override
    async def shutdown(self) -> None:
        if self.websocket is not None and not self.websocket.closed:
            await self.websocket.close()
        if hasattr(self, "site"):
            await self.site.stop()
        if hasattr(self, "runner"):
            await self.runner.cleanup()
        if hasattr(self, "_msg_send_stream"):
            await self._msg_send_stream.aclose()
            await self._msg_receive_stream.aclose()

    async def handle_response(self, request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        self.websocket = websocket
        await self._consume_websocket(websocket)
        return websocket

    async def _consume_websocket(self, websocket: web.WebSocketResponse) -> None:
        async for message in websocket:
            if message.type == aiohttp.WSMsgType.TEXT:
                await self._msg_send_stream.send(message)
            elif message.type == aiohttp.WSMsgType.ERROR:
                break

    async def _handle_msg_receive(self) -> None:
        async with anyio.create_task_group() as task_group, self._msg_receive_stream:
            async for message in self._msg_receive_stream:
                task_group.start_soon(self.handle_ws_response, message)

    @abstractmethod
    async def handle_ws_response(self, message: aiohttp.WSMessage) -> None:
        """Handle one WebSocket frame."""


class WebSocketAdapter(Adapter[EventT, ConfigT], metaclass=ABCMeta):
    """WebSocket adapter supporting client and reverse-server modes."""

    websocket: web.WebSocketResponse | aiohttp.ClientWebSocketResponse | None = None
    session: aiohttp.ClientSession | None = None
    app: web.Application | None = None
    runner: web.AppRunner | None = None
    site: web.TCPSite | None = None

    adapter_type: Literal["ws", "reverse-ws"]
    host: str
    port: int
    url: str
    reconnect_interval: float = 3

    _msg_send_stream: MemoryObjectSendStream[aiohttp.WSMessage]
    _msg_receive_stream: MemoryObjectReceiveStream[aiohttp.WSMessage]

    @override
    async def startup(self) -> None:
        self.websocket = None
        self.session = None
        self.app = None
        self.runner = None
        self.site = None
        if self.adapter_type == "ws":
            self.session = aiohttp.ClientSession()
        elif self.adapter_type == "reverse-ws":
            self.app = web.Application()
            self.app.add_routes([web.get(self.url, self.handle_reverse_ws_response)])
        else:  # pragma: no cover - Literal protects normal configuration
            raise ValueError('adapter_type must be "ws" or "reverse-ws"')
        self._msg_send_stream, self._msg_receive_stream = (
            anyio.create_memory_object_stream[aiohttp.WSMessage](
                max_buffer_size=float("inf")
            )
        )

    @override
    async def run(self) -> None:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(self._handle_msg_receive)
            try:
                if self.adapter_type == "ws":
                    await self._run_websocket_client()
                else:
                    await self._run_websocket_server()
            finally:
                await self._msg_send_stream.aclose()

    async def _run_websocket_client(self) -> None:
        while not self.should_stop:
            try:
                await self.websocket_connect()
            except aiohttp.ClientError:
                logger.exception("WebSocket connection error")
            if self.should_stop:
                return
            with anyio.move_on_after(self.reconnect_interval):
                await self.wait_stopped()

    async def _run_websocket_server(self) -> None:
        if self.app is None:
            raise RuntimeError("reverse WebSocket application is not initialized")
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        logger.info(
            "Reverse WebSocket listening",
            host=self.host,
            port=self.port,
            path=self.url,
        )
        await self.wait_stopped()

    @override
    async def shutdown(self) -> None:
        if self.websocket is not None and not self.websocket.closed:
            await self.websocket.close()
        self.websocket = None
        if self.site is not None:
            await self.site.stop()
        self.site = None
        if self.runner is not None:
            await self.runner.cleanup()
        self.runner = None
        if self.session is not None:
            await self.session.close()
        self.session = None
        if hasattr(self, "_msg_send_stream"):
            await self._msg_send_stream.aclose()
            await self._msg_receive_stream.aclose()

    async def handle_reverse_ws_response(
        self,
        request: web.Request,
    ) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse()
        await websocket.prepare(request)
        self.websocket = websocket
        await self.reverse_ws_connection_hook()
        await self._consume_websocket(websocket)
        return websocket

    async def reverse_ws_connection_hook(self) -> None:
        logger.info("WebSocket connected")

    async def websocket_connect(self) -> None:
        if self.session is None:
            raise RuntimeError("WebSocket client session is not initialized")
        async with self.session.ws_connect(
            f"ws://{self.host}:{self.port}{self.url}"
        ) as websocket:
            self.websocket = websocket
            await self._consume_websocket(websocket)

    async def handle_websocket(self) -> None:
        websocket = self.websocket
        if websocket is None or websocket.closed:
            return
        await self._consume_websocket(websocket)

    async def _consume_websocket(
        self,
        websocket: web.WebSocketResponse | aiohttp.ClientWebSocketResponse,
    ) -> None:
        async for message in websocket:
            await checkpoint()
            await self._msg_send_stream.send(message)
        if not self.should_stop:
            logger.warning("WebSocket connection closed")

    async def _handle_msg_receive(self) -> None:
        async with anyio.create_task_group() as task_group, self._msg_receive_stream:
            async for message in self._msg_receive_stream:
                task_group.start_soon(self.handle_websocket_msg, message)

    @abstractmethod
    async def handle_websocket_msg(self, message: aiohttp.WSMessage) -> None:
        raise NotImplementedError
