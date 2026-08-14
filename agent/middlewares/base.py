from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from contextlib import suppress
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
    from collections.abc import Awaitable, Callable

    from anyio.abc import TaskGroup
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

BatchT = TypeVar("BatchT")

_DEFAULT_RETRY_BACKOFF = 5.0
_GLOBAL_BACKGROUND_SESSION_LIMITER = anyio.CapacityLimiter(4)


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
        coalesce_seconds: float = 0.18,
        max_concurrent_sessions: int = 4,
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
        self._coalesce_seconds = max(0.0, coalesce_seconds)
        self._session_limiter = anyio.CapacityLimiter(
            max(1, max_concurrent_sessions)
        )

        self._pending_batches: LRUCache[str, deque[BatchT]] = LRUCache(
            maxsize=max_sessions
        )
        self._retry_attempts: LRUCache[str, int] = LRUCache(maxsize=max_sessions)

        self._scheduled_sessions: set[str] = set()
        self._delayed_retry_sessions: set[str] = set()
        self._enqueue_versions: dict[str, int] = {}

        self._start_lock = anyio.Lock()
        self._task_group: TaskGroup | None = None
        self._send_stream: MemoryObjectSendStream[str] | None = None
        self._receive_stream: MemoryObjectReceiveStream[str] | None = None

    async def aclose(self) -> None:
        task_group = self._task_group
        send_stream = self._send_stream
        receive_stream = self._receive_stream

        self._task_group = None
        self._send_stream = None
        self._receive_stream = None

        if task_group is not None:
            task_group.cancel_scope.cancel()
            try:
                await task_group.__aexit__(None, None, None)
            except* Exception as exc_group:
                self._logger.exception(
                    "Failed to close %s background tasks",
                    self._worker_name,
                    exc_info=exc_group,
                )

        for stream in (send_stream, receive_stream):
            if stream is None:
                continue
            with suppress(anyio.BrokenResourceError, anyio.ClosedResourceError):
                await stream.aclose()

        self._pending_batches.clear()
        self._scheduled_sessions.clear()
        self._delayed_retry_sessions.clear()
        self._retry_attempts.clear()
        self._enqueue_versions.clear()

        await self.on_close()

    async def _run_background_task(
        self,
        task_factory: Callable[..., Awaitable[None]],
        *args: Any,
    ) -> None:
        try:
            await task_factory(*args)
        except Exception:
            self._logger.exception("Failed to run %s background task", self._job_name)

    def _start_background_task(
        self,
        task_factory: Callable[..., Awaitable[None]],
        *args: Any,
    ) -> None:
        if self._task_group is None:
            msg = f"{self._worker_name} background task group is not running"
            raise RuntimeError(msg)
        self._task_group.start_soon(self._run_background_task, task_factory, *args)

    async def _ensure_service(self) -> None:
        if self._send_stream is not None:
            return

        async with self._start_lock:
            if self._send_stream is not None:
                return

            send_stream, receive_stream = anyio.create_memory_object_stream[str](
                self._queue_size
            )
            task_group = anyio.create_task_group()
            await task_group.__aenter__()
            try:
                self._send_stream = send_stream
                self._receive_stream = receive_stream
                self._task_group = task_group
                task_group.start_soon(self._save_worker, receive_stream)
            except BaseException:
                self._send_stream = None
                self._receive_stream = None
                self._task_group = None
                task_group.cancel_scope.cancel()
                try:
                    await task_group.__aexit__(None, None, None)
                finally:
                    await send_stream.aclose()
                    await receive_stream.aclose()
                raise

    async def _enqueue_batch(self, session_id: str, batch: BatchT) -> None:
        await self._ensure_service()
        if session_id not in self._pending_batches:
            self._pending_batches[session_id] = deque(
                maxlen=self._max_batches_per_session
            )
        self._pending_batches[session_id].append(batch)
        self._enqueue_versions[session_id] = self._enqueue_versions.get(session_id, 0) + 1
        await self._queue_job(session_id)

    async def _save_worker(
        self,
        receive_stream: MemoryObjectReceiveStream[str],
    ) -> None:
        async with receive_stream:
            while self._receive_stream is not None:
                try:
                    async for session_id in receive_stream:
                        # The receiver only dispatches.  Slow model analysis for
                        # one QQ session must never serialize every other group.
                        self._start_background_task(
                            self._process_session_job, session_id
                        )
                except Exception:
                    if self._receive_stream is None:
                        break
                    self._logger.exception(
                        "%s worker crashed, restarting in %s s",
                        self._worker_name,
                        _DEFAULT_RETRY_BACKOFF,
                    )
                    await anyio.sleep(_DEFAULT_RETRY_BACKOFF)

    async def _process_session_job(self, session_id: str) -> None:
        async with self._session_limiter, _GLOBAL_BACKGROUND_SESSION_LIMITER:
            # A small debounce window turns bursts of one-message events into a
            # single model request without adding latency to the reply path.
            if self._coalesce_seconds:
                await anyio.sleep(self._coalesce_seconds)
            start_version = self._enqueue_versions.get(session_id, 0)
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
                return

            self._retry_attempts.pop(session_id, None)
            received_during_processing = (
                self._enqueue_versions.get(session_id, 0) != start_version
            )
            if (progressed or received_during_processing) and self._pending_batches.get(
                session_id
            ):
                await self._queue_job(session_id)

    async def _drain_session(self, session_id: str) -> tuple[bool, bool]:
        pending_batches = self._pending_batches.get(session_id)
        if not pending_batches:
            return False, False
        batches = tuple(list(pending_batches)[: self._max_batch_window_size])
        if not batches:
            return False, False

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
            return False, True
        consumed_batches = min(max(result.consumed_batches, 0), len(batches))
        if consumed_batches <= 0:
            return False, False
        self._pop_processed_batches(session_id, consumed_batches)
        return True, False

    async def _queue_job(self, session_id: str) -> None:
        if (
            self._send_stream is None
            or self._task_group is None
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
            self._start_background_task(__send_job, self._send_stream)
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
        if session_id in self._delayed_retry_sessions or self._task_group is None:
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
            self._start_background_task(__retry_job)
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
