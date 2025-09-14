from __future__ import annotations

import itertools
import multiprocessing as mp
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
from anyio import to_thread

from .worker import Job, Result, worker_entry

if TYPE_CHECKING:
    from multiprocessing.context import SpawnContext


class PipelineProcess:
    """Long-running worker process bound to a single configuration."""

    _ids = itertools.count(1)
    _ctx: SpawnContext | None = None
    _ctx_lock: anyio.Lock = anyio.Lock()

    def __init__(
        self,
        cfg_path: str,
        ctx: SpawnContext,
        *,
        name: str | None = None,
        heartbeat: float = 30.0,
        other_chains: list[Any] | None = None,
    ) -> None:
        """Spawns a new worker process and prepares communication channels."""

        self._task_q: mp.Queue = ctx.Queue(maxsize=128)
        self._res_q: mp.Queue = ctx.Queue(maxsize=128)
        self._proc = ctx.Process(
            target=worker_entry,
            args=(cfg_path, self._task_q, self._res_q),
            kwargs={"other_chains": other_chains or []},
            name=name or f"worker-{Path(cfg_path).stem}",
            daemon=True,
        )
        self._proc.start()
        self._heartbeat = heartbeat
        self._last_pong = time.monotonic()

    async def ainvoke(
        self,
        input: Any,  # noqa: A002
        *,
        config: dict | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Submits an asynchronous invocation to the worker and returns the result."""
        job_id = next(self._ids)
        await to_thread.run_sync(self._task_q.put, Job("invoke", job_id, input, config))
        with anyio.fail_after(timeout or 90):
            res: Result = await to_thread.run_sync(self._res_q.get)
        if not res.ok:
            raise RuntimeError(f"Worker error:\n{res.data}")
        return res.data

    async def health_check(self) -> bool:
        """Sends a heartbeat to the worker and returns True on success."""
        await to_thread.run_sync(self._task_q.put, Job("heartbeat", -1))
        try:
            with anyio.fail_after(5):
                res: Result = await to_thread.run_sync(self._res_q.get)
            self._last_pong = time.monotonic()
        except TimeoutError:
            return False
        else:
            return res.ok

    async def aclose(self) -> None:
        """Gracefully terminates the worker process."""
        await to_thread.run_sync(self._task_q.put, Job("quit", -1))
        await to_thread.run_sync(self._proc.join, 5)


async def run_demo() -> None:
    """Demonstrates concurrent use of two independent pipelines."""
    from datetime import datetime  # noqa: PLC0415
    from zoneinfo import ZoneInfo  # noqa: PLC0415

    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    pipe_a = PipelineProcess("private_config.toml", ctx, name="A")
    r = await pipe_a.ainvoke(
        {
            "input": "你好",
            "time": datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime(
                "%Y年%m月%d日 %H时%M分"
            ),
            "optional_prompt": "",
        },
        config={"configurable": {"session_id": "session_id"}},
    )
    print("A:", r)
    await pipe_a.aclose()


if __name__ == "__main__":
    anyio.run(run_demo)
