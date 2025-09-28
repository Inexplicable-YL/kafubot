from __future__ import annotations

import multiprocessing as mp
import traceback
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from threading import Event, RLock, Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NotRequired,
    TypedDict,
    cast,
)
from typing_extensions import override

import anyio
from anyio import CapacityLimiter, to_thread
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from multiprocessing.connection import Connection
    from multiprocessing.context import BaseContext

    from langchain_core.runnables.config import RunnableConfig


class MsgInvoke(TypedDict):
    type: Literal["invoke"]
    id: str
    input: Any
    config: RunnableConfig | None
    kwargs: Mapping[str, Any]


class MsgAInvoke(TypedDict):
    type: Literal["ainvoke"]
    id: str
    input: Any
    config: RunnableConfig | None
    kwargs: Mapping[str, Any]


class MsgShutdown(TypedDict):
    type: Literal["shutdown"]


class MsgResultOk(TypedDict):
    id: str
    ok: Literal[True]
    result: Any


class MsgResultErr(TypedDict):
    id: str
    ok: Literal[False]
    error: str
    tb: NotRequired[str]


@dataclass
class _Waiter:
    event: Event = field(default_factory=Event)
    ok: bool = False
    result: Any = None
    error: str | None = None
    tb: str | None = None


class _RemoteError(Exception):
    def __init__(self, message: str, tb: str | None = None) -> None:
        super().__init__(message)
        self.tb = tb

    def __str__(self) -> str:
        base = super().__str__()
        if self.tb:
            return f"{base}\nRemote traceback:\n{self.tb}"
        return base


async def _worker_loop(
    conn: Connection, build_func: Callable[[], Runnable[Input, Output]]
) -> None:
    runnable = build_func()
    invoke_gate = CapacityLimiter(1)
    async with anyio.create_task_group() as tg:

        async def handle_messages() -> None:
            while True:
                msg: dict[str, Any] = await to_thread.run_sync(conn.recv)
                typ = msg.get("type")
                if typ == "shutdown":
                    await to_thread.run_sync(tg.cancel_scope.cancel)
                    break
                if typ == "ainvoke":
                    msg = cast("MsgAInvoke", msg)
                    rid = msg["id"]
                    data = msg["input"]
                    cfg = msg.get("config")
                    kwargs = msg.get("kwargs", {})
                    tg.start_soon(run_ainvoke, rid, data, cfg, kwargs)
                elif typ == "invoke":
                    msg = cast("MsgInvoke", msg)
                    rid = msg["id"]
                    data = msg["input"]
                    cfg = msg.get("config")
                    kwargs = msg.get("kwargs", {})
                    tg.start_soon(run_invoke, rid, data, cfg, kwargs)

        async def run_ainvoke(
            rid: str, data: Input, cfg: RunnableConfig | None, kwargs: Mapping[str, Any]
        ) -> None:
            try:
                res = await runnable.ainvoke(data, config=cfg, **dict(kwargs))
                await to_thread.run_sync(
                    conn.send, {"id": rid, "ok": True, "result": res}
                )
            except BaseException as e:
                tb = traceback.format_exc()
                await to_thread.run_sync(
                    conn.send, {"id": rid, "ok": False, "error": repr(e), "tb": tb}
                )

        async def run_invoke(
            rid: str, data: Input, cfg: RunnableConfig | None, kwargs: Mapping[str, Any]
        ) -> None:
            async with invoke_gate:
                try:
                    res = await to_thread.run_sync(
                        lambda: runnable.invoke(data, config=cfg, **dict(kwargs))
                    )
                    await to_thread.run_sync(
                        conn.send, {"id": rid, "ok": True, "result": res}
                    )
                except BaseException as e:
                    tb = traceback.format_exc()
                    await to_thread.run_sync(
                        conn.send, {"id": rid, "ok": False, "error": repr(e), "tb": tb}
                    )

        await handle_messages()
    with suppress(Exception):
        close_coro = getattr(runnable, "aclose", None)
        if callable(close_coro):
            await close_coro()
        else:
            close_fn = getattr(runnable, "close", None)
            if callable(close_fn):
                close_fn()


def _worker_main(
    conn: Connection, build_func: Callable[[], Runnable[Input, Output]]
) -> None:
    anyio.run(_worker_loop, conn, build_func)


class PipelineProcess(Runnable[Input, Output], Generic[Input, Output]):
    _ctx: BaseContext
    _conn: Connection
    _send_lock: RLock
    _waiters_lock: RLock
    _proc: mp.Process
    _waiters: dict[str, _Waiter]
    _invoke_lock: RLock
    _closed: bool
    _reader: Thread

    def __init__(
        self,
        build_func: Callable[[], Runnable[Input, Output]],
        mp_start_method: str = "spawn",
    ) -> None:
        self._ctx = mp.get_context(mp_start_method)
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        self._conn = parent_conn
        self._send_lock = RLock()
        self._waiters_lock = RLock()
        self._proc = self._ctx.Process(
            target=_worker_main, args=(child_conn, build_func), daemon=True
        )
        self._waiters = {}
        self._invoke_lock = RLock()
        self._closed = False
        self._reader = Thread(
            target=self._reader_loop,
            name=f"PipelineProcess-Reader-{id(self)}",
            daemon=True,
        )
        self._proc.start()
        self._reader.start()

    def _reader_loop(self) -> None:
        try:
            while True:
                msg: MsgResultOk | MsgResultErr = self._conn.recv()
                rid = msg.get("id")
                with self._waiters_lock:
                    w = self._waiters.get(rid)
                if not w:
                    continue
                w.ok = bool(msg.get("ok"))
                if w.ok:
                    w.result = msg.get("result")
                else:
                    w.error = msg.get("error")
                    w.tb = msg.get("tb")
                w.event.set()
        except EOFError:
            pass
        except OSError:
            pass
        finally:
            with self._waiters_lock:
                for w in list(self._waiters.values()):
                    if not w.event.is_set():
                        w.error = "connection closed"
                        w.event.set()

    def _send(self, payload: MsgInvoke | MsgAInvoke | MsgShutdown) -> None:
        with self._send_lock:
            self._conn.send(payload)

    @override
    def invoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        if self._closed:
            raise RuntimeError("PipelineProcess closed")
        rid = uuid.uuid4().hex
        w = _Waiter()
        with self._waiters_lock:
            self._waiters[rid] = w
        with self._invoke_lock:
            payload: MsgInvoke = {
                "type": "invoke",
                "id": rid,
                "input": input,
                "config": config,
                "kwargs": dict(kwargs),
            }
            self._send(payload)
            w.event.wait()
        with self._waiters_lock:
            del self._waiters[rid]
        if not w.ok:
            raise _RemoteError(w.error or "remote error", w.tb)
        return w.result

    @override
    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        if self._closed:
            raise RuntimeError("PipelineProcess closed")
        rid = uuid.uuid4().hex
        w = _Waiter()
        with self._waiters_lock:
            self._waiters[rid] = w
        payload: MsgAInvoke = {
            "type": "ainvoke",
            "id": rid,
            "input": input,
            "config": config,
            "kwargs": dict(kwargs),
        }
        self._send(payload)
        await to_thread.run_sync(w.event.wait)
        with self._waiters_lock:
            del self._waiters[rid]
        if not w.ok:
            raise _RemoteError(w.error or "remote error", w.tb)
        return w.result

    def close(self) -> None:
        if self._closed:
            return
        with suppress(Exception):
            self._send({"type": "shutdown"})
        with suppress(Exception):
            self._conn.close()
        with suppress(Exception):
            if self._proc.is_alive():
                self._proc.join(timeout=5)
        self._closed = True

    async def aclose(self) -> None:
        await to_thread.run_sync(self.close)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
