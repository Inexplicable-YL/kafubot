from __future__ import annotations

import itertools
import multiprocessing as mp
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
from anyio import from_thread, to_thread

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
        *,
        name: str | None = None,
        heartbeat: float = 30.0,
        other_chains: list[Any] | None = None,
    ) -> None:
        """Spawns a new worker process and prepares communication channels."""

        async def _ensure_ctx() -> None:
            async with PipelineProcess._ctx_lock:
                if PipelineProcess._ctx is None:
                    mp.set_start_method("spawn", force=True)
                    PipelineProcess._ctx = mp.get_context("spawn")

        try:
            from_thread.run(_ensure_ctx)  # inside an event loop
        except RuntimeError:
            anyio.run(_ensure_ctx)  # outside an event loop

        ctx = PipelineProcess._ctx  # guaranteed non-None

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
        await to_thread.run_sync(self._task_q.put, Job("heartbeat", None))
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
        await to_thread.run_sync(self._task_q.put, Job("quit", None))
        self._proc.join(timeout=5)


async def run_demo() -> None:
    """Demonstrates concurrent use of two independent pipelines."""
    pipe_a = PipelineProcess("cfg_a.toml", name="A")
    pipe_b = PipelineProcess("cfg_b.toml", name="B")
    async with anyio.create_task_group() as tg:
        tg.start_soon(lambda: pipe_a.ainvoke({"text": "hello"}))
        tg.start_soon(lambda: pipe_b.ainvoke({"text": "world"}))


if __name__ == "__main__":
    anyio.run(run_demo)
