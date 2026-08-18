from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

if TYPE_CHECKING:
    from anyio.abc import TaskGroup
    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

logger = logging.getLogger(__name__)
_DEFAULT_PATH = Path(".logs/social_control.jsonl")


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


class _TelemetryWriter:
    """Bounded, batched JSONL writer that never puts disk I/O on a QQ turn."""

    def __init__(self, queue_size: int = 512, batch_size: int = 48) -> None:
        self._queue_size = queue_size
        self._batch_size = batch_size
        self._start_lock = anyio.Lock()
        self._task_group: TaskGroup | None = None
        self._send_stream: MemoryObjectSendStream[dict[str, Any]] | None = None
        self._receive_stream: MemoryObjectReceiveStream[dict[str, Any]] | None = None

    async def _ensure_service(self) -> None:
        if self._send_stream is not None:
            return
        async with self._start_lock:
            if self._send_stream is not None:
                return
            send_stream, receive_stream = anyio.create_memory_object_stream[
                dict[str, Any]
            ](self._queue_size)
            task_group = anyio.create_task_group()
            await task_group.__aenter__()
            self._send_stream = send_stream
            self._receive_stream = receive_stream
            self._task_group = task_group
            task_group.start_soon(self._worker, receive_stream)

    async def emit(self, record: dict[str, Any]) -> None:
        await self._ensure_service()
        if self._send_stream is None:
            return
        try:
            self._send_stream.send_nowait(record)
        except anyio.WouldBlock:
            logger.warning(
                "social telemetry queue full; dropping event=%s", record.get("event")
            )
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            return

    async def _worker(
        self, receive_stream: MemoryObjectReceiveStream[dict[str, Any]]
    ) -> None:
        async with receive_stream:
            async for first_record in receive_stream:
                records = [first_record]
                while len(records) < self._batch_size:
                    try:
                        records.append(receive_stream.receive_nowait())
                    except anyio.WouldBlock:
                        break
                    except anyio.EndOfStream:
                        break
                path = Path(os.getenv("SOCIAL_TELEMETRY_PATH", str(_DEFAULT_PATH)))
                lines = "".join(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        default=_json_default,
                        separators=(",", ":"),
                    )
                    + "\n"
                    for record in records
                )
                try:
                    await anyio.Path(path.parent).mkdir(parents=True, exist_ok=True)
                    async with await anyio.open_file(
                        path, "a", encoding="utf-8"
                    ) as stream:
                        await stream.write(lines)
                except Exception:
                    logger.exception("Failed to append social telemetry batch")

    async def aclose(self) -> None:
        send_stream, self._send_stream = self._send_stream, None
        receive_stream, self._receive_stream = self._receive_stream, None
        task_group, self._task_group = self._task_group, None
        if send_stream is not None:
            await send_stream.aclose()
        if task_group is not None:
            await task_group.__aexit__(None, None, None)
        if receive_stream is not None:
            await receive_stream.aclose()


_WRITER = _TelemetryWriter()


async def log_social_event(event: str, **payload: Any) -> None:
    """Queue replayable control telemetry without waiting for serialization/disk."""

    if os.getenv("SOCIAL_TELEMETRY_ENABLED", "1") == "0":
        return
    await _WRITER.emit(
        {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "event": event,
            **payload,
        }
    )


async def close_social_telemetry() -> None:
    await _WRITER.aclose()


__all__ = ["close_social_telemetry", "log_social_event"]
