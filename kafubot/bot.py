from __future__ import annotations

import signal
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import anyio

if TYPE_CHECKING:
    from anyio.abc import ObjectReceiveStream

    from kafubot.event import Event

from kafubot.adapters import get_adapter_class
from kafubot.adapters.cqhttp.event import GroupMessageEvent, PrivateMessageEvent
from kafubot.config import AppConfig, load_config
from kafubot.log import configure_logging, logger
from kafubot.runtimes import AgentRuntime, create_agent_runtime

ShutdownCallback = Callable[[], Awaitable[None]]
ChatMessageEvent: TypeAlias = GroupMessageEvent | PrivateMessageEvent


class Bot:
    """Protocol lifecycle plus one unified social-agent runtime."""

    def __init__(self, config_file: str | Path = "config.toml") -> None:
        self.config_file = Path(config_file)
        self.config: AppConfig = load_config(self.config_file)
        configure_logging(
            self.config.bot.log.level,
            self.config.bot.log.verbose_exception,
        )
        self.agent_runtime: AgentRuntime = create_agent_runtime(self.config.agent)
        self._shutdown_callbacks: list[ShutdownCallback] = [self.agent_runtime.aclose]
        self._should_exit = anyio.Event()
        self._event_send, self._event_receive = anyio.create_memory_object_stream[
            ChatMessageEvent
        ](max_buffer_size=self.config.bot.event_queue_size)
        adapter_class = get_adapter_class(self.config.bot.adapter)
        self.adapter = adapter_class(self)

    def run(self) -> None:
        with suppress(KeyboardInterrupt):  # pragma: no cover - terminal behavior
            anyio.run(self.arun)

    async def arun(self) -> None:
        logger.info("Starting KafuBot", adapter=self.adapter.name)
        try:
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(self._run_adapter)
                task_group.start_soon(self._watch_signals)
                task_group.start_soon(self.agent_runtime.run)
                for _ in range(self.config.bot.event_workers):
                    task_group.start_soon(
                        self._event_worker,
                        self._event_receive.clone(),
                    )
                await self._should_exit.wait()
                self.adapter.stop()
                task_group.cancel_scope.cancel()
        finally:
            with anyio.CancelScope(shield=True):
                await self._close_services()
                await self._event_send.aclose()
                await self._event_receive.aclose()
            logger.info("KafuBot stopped")

    def shutdown(self) -> None:
        self._should_exit.set()

    async def submit_event(self, event: Event) -> None:
        if isinstance(event, (GroupMessageEvent, PrivateMessageEvent)):
            await self._event_send.send(event)

    async def _run_adapter(self) -> None:
        await self.adapter.safe_run()
        if not self._should_exit.is_set():
            logger.error("CQHTTP adapter exhausted retries")
            self.shutdown()

    async def _event_worker(
        self,
        receive: ObjectReceiveStream[ChatMessageEvent],
    ) -> None:
        async with receive:
            async for event in receive:
                try:
                    logger.debug("Handling chat event", event_name=event.event_name)
                    await self.agent_runtime.handle(event)
                except anyio.get_cancelled_exc_class():
                    raise
                except Exception:
                    logger.exception(
                        "Unhandled event error",
                        event_name=getattr(
                            event,
                            "event_name",
                            type(event).__name__,
                        ),
                    )

    async def _watch_signals(self) -> None:
        try:
            with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
                async for received_signal in signals:
                    logger.info("Received shutdown signal", signal=received_signal.name)
                    self.shutdown()
                    return
        except (NotImplementedError, RuntimeError):  # pragma: no cover - Windows
            await anyio.sleep_forever()

    async def _close_services(self) -> None:
        for callback in reversed(self._shutdown_callbacks):
            try:
                await callback()
            except Exception:
                logger.exception("Service shutdown failed", callback=repr(callback))
        self._shutdown_callbacks.clear()
