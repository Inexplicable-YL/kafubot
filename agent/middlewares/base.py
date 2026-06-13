from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    NotRequired,
    TypedDict,
    TypeVar,
)

import anyio
from cachetools import LRUCache
from langchain.agents.middleware import AgentMiddleware

from agent.base import ManagerContext, ManagerState

if TYPE_CHECKING:
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

BatchT = TypeVar("BatchT")

_DEFAULT_RETRY_BACKOFF = 5.0


class LoggerConfig(TypedDict):
    worker_name: NotRequired[str]
    job_name: NotRequired[str]
    process_failure_log: NotRequired[str]
    retry_exhausted_label: NotRequired[str | None]


@dataclass(slots=True, frozen=True)
class ProcessResult:
    """Business result returned by subclasses for one batch window."""

    consumed_batches: int = 0
    failed: bool = False


# `None` -> consumed_batches=0, failed=False
# `int` -> consumed_batches=<int>, failed=False
# `tuple[int, bool]` -> consumed_batches, failed
SessionProcessOutput = int | tuple[int, bool] | ProcessResult | None


class BaseDaemonMiddleware(
    AgentMiddleware[ManagerState, ManagerContext],
    ABC,
    Generic[BatchT],
):
    """Shared session-scoped queue/worker/retry system for background middlewares."""

    state_schema = ManagerState

    def __init__(
        self,
        *,
        max_sessions: int,
        max_retries: int,
        max_batch_window_size: int,
        logger_config: LoggerConfig | None = None,
    ) -> None:
        super().__init__()
        if logger_config is None:
            logger_config = {}
        self.max_retries = max_retries

        self._logger = logging.getLogger(self.__class__.__module__)
        self._worker_name = logger_config.get("worker_name", self.__class__.__name__)
        self._job_name = logger_config.get("job_name", self._worker_name + " job")
        self._process_failure_log = logger_config.get(
            "process_failure_log", f"Failed to process {self._job_name}"
        )
        self._retry_exhausted_label = (
            logger_config.get("retry_exhausted_label") or self._worker_name
        )

        self._queue_size = max(64, max_sessions * 2)
        self._max_batches_per_session = max(16, max_sessions * 2)
        self._max_batch_window_size = max(1, max_batch_window_size)

        self._pending_batches: LRUCache[str, deque[BatchT]] = LRUCache(
            maxsize=max_sessions
        )
        self._retry_attempts: LRUCache[str, int] = LRUCache(maxsize=max_sessions)

        self._scheduled_sessions: set[str] = set()
        self._delayed_retry_sessions: set[str] = set()

        self._start_lock = anyio.Lock()
        self._worker_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._send_stream: MemoryObjectSendStream[str] | None = None
        self._receive_stream: MemoryObjectReceiveStream[str] | None = None

    async def aclose(self) -> None:
        if self._send_stream is not None:
            await self._send_stream.aclose()
        if self._receive_stream is not None:
            await self._receive_stream.aclose()
        tasks = list(self._background_tasks)
        if self._worker_task is not None:
            tasks.append(self._worker_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._send_stream = None
        self._receive_stream = None
        self._worker_task = None
        self._background_tasks.clear()

        self._pending_batches.clear()
        self._scheduled_sessions.clear()
        self._delayed_retry_sessions.clear()
        self._retry_attempts.clear()

        await self.on_close()

    def _clear_worker_task(self, task: asyncio.Task[None]) -> None:
        if self._worker_task is task:
            self._worker_task = None

    def _start_background_task(
        self,
        coro: Any,
    ) -> asyncio.Task[None]:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _ensure_service(self) -> None:
        if self._send_stream is not None:
            return

        async with self._start_lock:
            if self._send_stream is not None:
                return

            send_stream, receive_stream = anyio.create_memory_object_stream[str](
                self._queue_size
            )
            self._send_stream = send_stream
            self._receive_stream = receive_stream
            worker_task = asyncio.create_task(self._save_worker(receive_stream))
            worker_task.add_done_callback(self._clear_worker_task)
            self._worker_task = worker_task

    async def _enqueue_batch(self, session_id: str, batch: BatchT) -> None:
        await self._ensure_service()
        if session_id not in self._pending_batches:
            self._pending_batches[session_id] = deque(
                maxlen=self._max_batches_per_session
            )
        self._pending_batches[session_id].append(batch)
        await self._queue_job(session_id)

    async def _save_worker(
        self,
        receive_stream: MemoryObjectReceiveStream[str],
    ) -> None:
        async with receive_stream:
            while self._receive_stream is not None:
                try:
                    async for session_id in receive_stream:
                        try:
                            progressed, failed = await self._drain_session(session_id)
                        except Exception:
                            self._logger.exception(self._process_failure_log)
                            progressed = False
                            failed = True
                        finally:
                            self._scheduled_sessions.discard(session_id)

                        if failed:
                            await self._schedule_retry(session_id)
                            continue

                        self._retry_attempts.pop(session_id, None)
                        if progressed and self._pending_batches.get(session_id):
                            await self._queue_job(session_id)
                except Exception:
                    if self._receive_stream is None:
                        break
                    self._logger.exception(
                        "%s worker crashed, restarting in %s s",
                        self._worker_name,
                        _DEFAULT_RETRY_BACKOFF,
                    )
                    await anyio.sleep(_DEFAULT_RETRY_BACKOFF)

    async def _drain_session(self, session_id: str) -> tuple[bool, bool]:
        progressed = False
        while pending_batches := self._pending_batches.get(session_id):
            batches = tuple(list(pending_batches)[: self._max_batch_window_size])
            if not batches:
                continue

            result = await self.process_batches(session_id, batches)
            if result is None:
                result = ProcessResult()
            elif isinstance(result, int):
                result = ProcessResult(consumed_batches=result)
            elif isinstance(result, tuple):
                consumed_batches, failed = result
                result = ProcessResult(
                    consumed_batches=consumed_batches,
                    failed=failed,
                )
            elif not isinstance(result, ProcessResult):
                raise TypeError(
                    f"Invalid process result type: {type(result)}; expected int, tuple[int, bool], ProcessResult, or None"
                )

            if result.failed:
                return progressed, True
            consumed_batches = min(
                max(result.consumed_batches, 0),
                len(batches),
            )
            if consumed_batches <= 0:
                break
            self._pop_processed_batches(session_id, consumed_batches)
            progressed = True
        return progressed, False

    async def _queue_job(self, session_id: str) -> None:
        if (
            self._send_stream is None
            or self._worker_task is None
            or session_id in self._scheduled_sessions
        ):
            return

        self._scheduled_sessions.add(session_id)
        try:
            self._send_stream.send_nowait(session_id)
        except anyio.WouldBlock:
            pass
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            self._scheduled_sessions.discard(session_id)
            self._logger.exception("Failed to queue %s job", self._job_name)
            return
        else:
            return

        async def __send_job(
            send_stream: MemoryObjectSendStream[str],
        ) -> None:
            try:
                await send_stream.send(session_id)
            except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                self._scheduled_sessions.discard(session_id)
                self._logger.exception("Failed to send %s job", self._job_name)

        try:
            self._start_background_task(__send_job(self._send_stream))
        except RuntimeError:
            self._scheduled_sessions.discard(session_id)
            self._logger.exception("Failed to schedule %s job", self._job_name)

    async def _schedule_retry(self, session_id: str) -> None:
        if not self._pending_batches.get(session_id):
            self._retry_attempts.pop(session_id, None)
            return

        attempt = self._retry_attempts.get(session_id, 0) + 1
        if attempt > self.max_retries:
            self._logger.error(
                "%s retries exhausted for %s; pending batches=%s",
                self._retry_exhausted_label,
                session_id,
                len(self._pending_batches.get(session_id, ())),
            )
            self._pop_processed_batches(session_id, 1)
            self._retry_attempts.pop(session_id, None)
            return

        self._retry_attempts[session_id] = attempt
        if session_id in self._delayed_retry_sessions or self._worker_task is None:
            return

        async def __retry_job() -> None:
            try:
                await anyio.sleep(_DEFAULT_RETRY_BACKOFF * attempt)
            finally:
                self._delayed_retry_sessions.discard(session_id)

            if not self._pending_batches.get(session_id):
                self._retry_attempts.pop(session_id, None)
                return

            await self._queue_job(session_id)

        self._delayed_retry_sessions.add(session_id)
        try:
            self._start_background_task(__retry_job())
        except RuntimeError:
            self._delayed_retry_sessions.discard(session_id)
            self._logger.exception("Failed to schedule %s retry", self._job_name)

    def _pop_processed_batches(self, session_id: str, batch_count: int) -> None:
        if batch_count <= 0:
            return

        pending_batches = self._pending_batches.get(session_id)
        if not pending_batches:
            return

        for _ in range(min(batch_count, len(pending_batches))):
            pending_batches.popleft()
        if not pending_batches:
            self._pending_batches.pop(session_id, None)

    @abstractmethod
    async def process_batches(
        self,
        session_id: str,
        batches: tuple[BatchT, ...],
    ) -> SessionProcessOutput:
        """Process one session batch window using business logic only."""
        raise NotImplementedError

    async def on_close(self) -> None:
        """Hook for subclasses to clear caches or references on close."""


__all__ = [
    "BaseDaemonMiddleware",
    "LoggerConfig",
    "ProcessResult",
]
