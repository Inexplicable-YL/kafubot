from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast
from typing_extensions import override

import anyio
import pytest
from anyio.lowlevel import checkpoint

from kafubot.adapters import Adapter, get_adapter_class, register_adapter
from kafubot.adapters.cqhttp import CQHTTPAdapter
from kafubot.adapters.utils import (
    HttpClientAdapter,
    HttpServerAdapter,
    PollingAdapter,
    WebSocketAdapter,
    WebSocketClientAdapter,
    WebSocketServerAdapter,
)
from kafubot.config import AppConfig, BotConfig, ConfigModel
from kafubot.exceptions import GetEventTimeout, MockApiException
from kafubot.protocol import Event

if TYPE_CHECKING:
    from kafubot.bot import Bot


class DummyConfig(ConfigModel):
    __config_name__: ClassVar[str] = "dummy"
    value: int = 1


class DummyEvent(Event):
    value: int

    @property
    @override
    def event_name(self) -> str:
        return "dummy"


class FakeBot:
    def __init__(self, *, retries: int = 0) -> None:
        self.config = AppConfig(
            bot=BotConfig(adapter="dummy", adapter_max_retries=retries),
            adapter={"dummy": {"value": 7}},
        )
        self._should_exit = anyio.Event()
        self.received: list[Event] = []

    async def submit_event(self, event: Event) -> None:
        self.received.append(event)


class DummyAdapter(Adapter[DummyEvent, DummyConfig]):
    name = "dummy"
    Config = DummyConfig

    def __init__(self, bot: Bot) -> None:
        super().__init__(bot)
        self.api_calls: list[tuple[str, dict[str, Any]]] = []

    @override
    async def run(self) -> None:
        await self.wait_stopped()

    @override
    async def _call_api(self, api: str, **params: Any) -> Any:
        self.api_calls.append((api, params))
        return {"api": api, **params}

    @override
    async def send(self, event: Event, message: Any, **kwargs: Any) -> Any:
        return event, message, kwargs


def make_dummy_adapter(*, retries: int = 0) -> tuple[DummyAdapter, FakeBot]:
    bot = FakeBot(retries=retries)
    return DummyAdapter(cast("Bot", bot)), bot


def test_adapter_registry_and_config_binding() -> None:
    register_adapter(DummyAdapter)
    adapter, _ = make_dummy_adapter()

    assert get_adapter_class("dummy") is DummyAdapter
    assert get_adapter_class("cqhttp") is CQHTTPAdapter
    assert issubclass(CQHTTPAdapter, WebSocketAdapter)
    assert adapter.config.value == 7


def test_all_sekaibot_adapter_utilities_are_available() -> None:
    assert all(
        issubclass(adapter_type, Adapter)
        for adapter_type in (
            PollingAdapter,
            HttpClientAdapter,
            WebSocketClientAdapter,
            HttpServerAdapter,
            WebSocketServerAdapter,
            WebSocketAdapter,
        )
    )


@pytest.mark.anyio
async def test_api_hooks_can_mutate_params_and_mock_results() -> None:
    adapter, _ = make_dummy_adapter()

    async def before(
        _adapter: Adapter[Any, Any],
        _api: str,
        params: dict[str, Any],
    ) -> None:
        params["injected"] = True

    async def after(
        _adapter: Adapter[Any, Any],
        _exception: Exception | None,
        _api: str,
        _params: dict[str, Any],
        _result: Any,
    ) -> None:
        raise MockApiException("mocked-after")

    DummyAdapter.calling_api_hook(before)
    DummyAdapter.called_api_hook(after)
    try:
        result = await adapter.call_api("demo", original=True)
    finally:
        DummyAdapter._calling_api_hooks.discard(before)
        DummyAdapter._called_api_hooks.discard(after)

    assert result == "mocked-after"
    assert adapter.api_calls == [
        ("demo", {"original": True, "injected": True})
    ]


@pytest.mark.anyio
async def test_calling_api_hook_can_skip_transport() -> None:
    adapter, _ = make_dummy_adapter()

    async def mock_before(
        _adapter: Adapter[Any, Any],
        _api: str,
        _params: dict[str, Any],
    ) -> None:
        raise MockApiException("mocked-before")

    DummyAdapter.calling_api_hook(mock_before)
    try:
        result = await adapter.call_api("demo")
    finally:
        DummyAdapter._calling_api_hooks.discard(mock_before)

    assert result == "mocked-before"
    assert adapter.api_calls == []


@pytest.mark.anyio
async def test_adapter_get_waits_for_its_future_native_events() -> None:
    adapter, bot = make_dummy_adapter()
    caught: list[DummyEvent] = []
    first = DummyEvent(adapter=adapter, value=1)
    second = DummyEvent(adapter=adapter, value=2)

    async def wait_for_event() -> None:
        caught.append(
            await adapter.get(
                lambda event: event.value == 2,
                event_type=DummyEvent,
                timeout=1,
            )
        )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(wait_for_event)
        while not adapter._event_waiters:
            await checkpoint()
        await adapter.handle_event(first)
        await adapter.handle_event(second)

    assert caught == [second]
    assert bot.received == [first, second]


@pytest.mark.anyio
async def test_adapter_get_raises_protocol_timeout() -> None:
    adapter, _ = make_dummy_adapter()

    with pytest.raises(GetEventTimeout):
        await adapter.get(timeout=0.01)


@pytest.mark.anyio
async def test_safe_run_retries_with_startup_and_shutdown() -> None:
    class RetryAdapter(DummyAdapter):
        def __init__(self, bot: Bot) -> None:
            super().__init__(bot)
            self.startups = 0
            self.runs = 0
            self.shutdowns = 0

        @override
        async def startup(self) -> None:
            self.startups += 1

        @override
        async def run(self) -> None:
            self.runs += 1
            if self.runs == 1:
                raise RuntimeError("retry")
            self.stop()

        @override
        async def shutdown(self) -> None:
            self.shutdowns += 1

        @override
        def retry_delay(self, retries: int) -> float:
            _ = retries
            return 0

    bot = FakeBot(retries=1)
    adapter = RetryAdapter(cast("Bot", bot))

    await adapter.safe_run()

    assert (adapter.startups, adapter.runs, adapter.shutdowns) == (2, 2, 2)
