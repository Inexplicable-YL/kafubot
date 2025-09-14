from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import anyio
from anyio import to_thread

if TYPE_CHECKING:
    from multiprocessing import Queue


@dataclass(slots=True)
class Job:
    """Message sent from the parent process."""

    op: Literal["invoke", "heartbeat", "quit"]
    job_id: int
    inp: Any | None = None
    invoke_cfg: dict[str, Any] | None = None


@dataclass(slots=True)
class Result:
    """Message sent back to the parent process."""

    job_id: int
    ok: bool
    data: Any


def worker_entry(
    cfg_path: str, task_q: Queue, result_q: Queue, *, other_chains: list[Any]
) -> None:
    """Spawn target that hosts a single pipeline instance."""
    cfg = Path(cfg_path).resolve()
    if not cfg.is_file():
        raise FileNotFoundError(cfg)

    from cogniweave import build_pipeline, init_config  # noqa: PLC0415

    init_config(_config_file=str(cfg))
    runnable = build_pipeline()
    for chain in other_chains:
        runnable |= chain

    async def handle_invoke(job: Job) -> None:
        """Runs the pipeline and returns the output."""
        try:
            with anyio.fail_after(60):
                out = await runnable.ainvoke(input=job.inp, config=job.invoke_cfg or {}) # type: ignore
            await to_thread.run_sync(result_q.put, Result(job.job_id, True, out))
        except BaseException as exc:
            tb = "".join(traceback.format_exception(exc))
            await to_thread.run_sync(result_q.put, Result(job.job_id, False, tb))

    async def serve() -> None:
        """Main service loop processing incoming jobs."""
        async with anyio.create_task_group() as tg:
            while True:
                job: Job = await to_thread.run_sync(task_q.get)
                if job.op == "quit":
                    break
                if job.op == "heartbeat":
                    await to_thread.run_sync(result_q.put, Result(-1, True, "pong"))
                    continue
                tg.start_soon(handle_invoke, job)

    anyio.run(serve)
