from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast, overload

import anyio
from pydantic import BaseModel

from kafubot.event import Event
from kafubot.exceptions import GetEventTimeout, MockApiException
from kafubot.log import logger
from kafubot.message import BuildMessageType, MessageSegment

if TYPE_CHECKING:
    from anyio.streams.memory import MemoryObjectSendStream

    from kafubot.bot import Bot

EventT = TypeVar("EventT", bound=Event)
RequestedEventT = TypeVar("RequestedEventT", bound=Event)
ConfigT = TypeVar("ConfigT", bound=BaseModel | None)
MessageSegmentT = TypeVar("MessageSegmentT", bound=MessageSegment[Any])
CallingAPIHook = Callable[
    ["Adapter[Any, Any]", str, dict[str, Any]],
    Any | Awaitable[Any],
]
CalledAPIHook = Callable[
    ["Adapter[Any, Any]", Exception | None, str, dict[str, Any], Any],
    Any | Awaitable[Any],
]
AdapterT = TypeVar("AdapterT", bound="Adapter[Any, Any]")

__all__ = [
    "Adapter",
    "CalledAPIHook",
    "CallingAPIHook",
    "ConfigT",
    "EventT",
    "get_adapter_class",
    "register_adapter",
]

_ADAPTERS: dict[str, type[Adapter[Any, Any]]] = {}


class Adapter(ABC, Generic[EventT, ConfigT]):
    """Extensible protocol adapter base, ported from SekaiBot."""

    name: str
    Config: type[BaseModel] | None = None

    _calling_api_hooks: ClassVar[set[CallingAPIHook]] = set()
    _called_api_hooks: ClassVar[set[CalledAPIHook]] = set()

    def __init__(self, bot: Bot) -> None:
        self.name = getattr(self, "name", self.__class__.__name__)
        self.bot = bot
        self._stopping = anyio.Event()
        self._event_waiters: set[MemoryObjectSendStream[Event]] = set()
        self._waiters_lock = anyio.Lock()

    @property
    def config(self) -> ConfigT:
        """Resolve this adapter's Pydantic config from the Bot configuration."""
        config_class = self.Config
        if config_class is None:
            return cast("ConfigT", None)
        config_name = str(
            getattr(config_class, "__config_name__", None) or self.name
        )
        adapter_config = self.bot.config.adapter
        raw_config = (
            adapter_config.get(config_name, {})
            if isinstance(adapter_config, Mapping)
            else getattr(adapter_config, config_name, {})
        )
        if isinstance(raw_config, config_class):
            return cast("ConfigT", raw_config)
        return cast("ConfigT", config_class.model_validate(raw_config))

    @property
    def should_stop(self) -> bool:
        return self._stopping.is_set() or self.bot._should_exit.is_set()

    def stop(self) -> None:
        self._stopping.set()
        for waiter in tuple(self._event_waiters):
            waiter.close()

    async def wait_stopped(self) -> None:
        await self._stopping.wait()

    async def safe_run(self) -> None:
        """Run with startup/shutdown isolation and configured retries."""
        retries = 0
        max_retries = self.bot.config.bot.adapter_max_retries
        while not self.should_stop:
            try:
                await self.startup()
                await self.run()
                if not self.should_stop:
                    raise RuntimeError("adapter stopped unexpectedly")
            except anyio.get_cancelled_exc_class():
                raise
            except Exception:
                logger.exception(
                    "Adapter failed",
                    adapter=self.__class__,
                    retries=retries,
                )
            finally:
                with anyio.CancelScope(shield=True):
                    try:
                        await self.shutdown()
                    except Exception:
                        logger.exception(
                            "Adapter shutdown failed",
                            adapter=self.__class__,
                        )

            if self.should_stop:
                break
            if retries >= max_retries:
                logger.warning(
                    "Adapter run failed after retries",
                    adapter_name=self.__class__.__name__,
                )
                break
            retries += 1
            logger.info(
                "Retrying adapter",
                adapter=self.__class__,
                retries=retries,
            )
            await anyio.sleep(self.retry_delay(retries))

    def retry_delay(self, retries: int) -> float:
        return float(min(30, max(1, retries)))

    async def startup(self) -> None:
        """Initialize adapter resources before each run attempt."""

    @abstractmethod
    async def run(self) -> None:
        raise NotImplementedError

    async def shutdown(self) -> None:
        """Release adapter resources after a run attempt."""

    @abstractmethod
    async def _call_api(self, api: str, **params: Any) -> Any:
        raise NotImplementedError

    async def call_api(self, api: str, **params: Any) -> Any:
        """Call an API through SekaiBot-compatible before/after hooks."""
        result: Any = None
        exception: Exception | None = None
        mocked_results: list[Any] = []

        async def run_calling_hook(hook: CallingAPIHook) -> None:
            try:
                hook_result = hook(self, api, params)
                if inspect.isawaitable(hook_result):
                    await hook_result
            except MockApiException as exc:
                mocked_results.append(exc.result)
            except Exception:
                logger.exception("Error while running CallingAPI hook")

        if self._calling_api_hooks:
            async with anyio.create_task_group() as task_group:
                for hook in tuple(self._calling_api_hooks):
                    task_group.start_soon(run_calling_hook, hook)
        if mocked_results:
            if len(mocked_results) > 1:
                logger.warning("Multiple hooks mocked an API result; using the first")
            result = mocked_results[0]
        else:
            try:
                result = await self._call_api(api, **params)
            except Exception as exc:
                exception = exc

        mocked_results = []

        async def run_called_hook(hook: CalledAPIHook) -> None:
            try:
                hook_result = hook(self, exception, api, params, result)
                if inspect.isawaitable(hook_result):
                    await hook_result
            except MockApiException as exc:
                mocked_results.append(exc.result)
            except Exception:
                logger.exception("Error while running CalledAPI hook")

        if self._called_api_hooks:
            async with anyio.create_task_group() as task_group:
                for hook in tuple(self._called_api_hooks):
                    task_group.start_soon(run_called_hook, hook)
        if mocked_results:
            if len(mocked_results) > 1:
                logger.warning("Multiple hooks mocked an API result; using the first")
            result = mocked_results[0]
            exception = None

        if exception is not None:
            raise exception
        return result

    @abstractmethod
    async def send(
        self,
        event: Event,
        message: BuildMessageType[MessageSegmentT],
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError

    async def handle_event(self, event: EventT) -> None:
        """Publish a native event to waiters, then hand it to the Bot ingress."""
        async with self._waiters_lock:
            waiters = tuple(self._event_waiters)
        for waiter in waiters:
            try:
                waiter.send_nowait(event)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                continue
        await self.bot.submit_event(event)

    @overload
    async def get(
        self,
        func: Callable[[EventT], bool | Awaitable[bool]] | None = None,
        *,
        event_type: None = None,
        max_try_times: int | None = None,
        timeout: float | None = None,
    ) -> EventT: ...

    @overload
    async def get(
        self,
        func: Callable[[RequestedEventT], bool | Awaitable[bool]] | None = None,
        *,
        event_type: type[RequestedEventT],
        max_try_times: int | None = None,
        timeout: float | None = None,
    ) -> RequestedEventT: ...

    async def get(
        self,
        func: Callable[[Any], bool | Awaitable[bool]] | None = None,
        *,
        event_type: type[Event] | None = None,
        max_try_times: int | None = None,
        timeout: float | None = None,
    ) -> Event:
        """Wait for a future event emitted by this adapter."""
        send_stream, receive_stream = anyio.create_memory_object_stream[Event](
            max_buffer_size=float("inf")
        )
        async with self._waiters_lock:
            self._event_waiters.add(send_stream)

        async def wait_for_match() -> Event:
            failures = 0
            async with receive_stream:
                async for event in receive_stream:
                    matched = event_type is None or isinstance(event, event_type)
                    if matched and func is not None:
                        predicate_result = func(event)
                        matched = bool(
                            await predicate_result
                            if inspect.isawaitable(predicate_result)
                            else predicate_result
                        )
                    if matched:
                        return event
                    failures += 1
                    if max_try_times is not None and failures > max_try_times:
                        break
            raise GetEventTimeout

        try:
            if timeout is None:
                return await wait_for_match()
            with anyio.fail_after(timeout):
                return await wait_for_match()
        except (TimeoutError, anyio.EndOfStream) as exc:
            raise GetEventTimeout from exc
        finally:
            async with self._waiters_lock:
                self._event_waiters.discard(send_stream)
            await send_stream.aclose()

    @classmethod
    def calling_api_hook(cls, func: CallingAPIHook) -> CallingAPIHook:
        cls._calling_api_hooks.add(func)
        return func

    @classmethod
    def called_api_hook(cls, func: CalledAPIHook) -> CalledAPIHook:
        cls._called_api_hooks.add(func)
        return func


def register_adapter(adapter_class: type[AdapterT]) -> type[AdapterT]:
    """Register an adapter class under its declared name."""
    if not issubclass(adapter_class, Adapter):
        raise TypeError("adapter_class must inherit Adapter")
    name = getattr(adapter_class, "name", adapter_class.__name__)
    _ADAPTERS[name] = adapter_class
    return adapter_class


def get_adapter_class(name: str) -> type[Adapter[Any, Any]]:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        raise LookupError(f"adapter is not registered: {name}") from exc


from .cqhttp import CQHTTPAdapter  # noqa: E402

register_adapter(CQHTTPAdapter)
__all__ += ["CQHTTPAdapter"]
