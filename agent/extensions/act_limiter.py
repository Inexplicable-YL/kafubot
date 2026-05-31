import asyncio
from datetime import UTC, datetime
from typing import Any, TypedDict

from langchain.agents.middleware import (
    AgentMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from sqlalchemy import Float, Index, Integer, Text, delete, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool

from agent.base import (
    ManagerContext,
    ManagerState,
    OutputMessage,
)


class _Base(DeclarativeBase):
    pass


class _Record(_Base):
    __tablename__ = "activate_limiter_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    __table_args__ = (Index("ix_session_time", "session_id", "timestamp"),)


class _Bucket(_Base):
    __tablename__ = "activate_limiter_buckets"

    session_id: Mapped[str] = mapped_column(Text, primary_key=True)
    available: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    last_update: Mapped[float] = mapped_column(Float, nullable=False)


class ActivateLimiterConfig(TypedDict):
    db_url: str
    act_limits: tuple[tuple[int, int], ...]
    rate_limit: tuple[float, float]


class ActivateLimiter:
    def __init__(
        self,
        db_url: str,
        act_limits: tuple[tuple[int, int], ...],
        rate_limit: tuple[float, float] | None = None,
    ) -> None:
        """
        Args:
            db_url: 数据库连接字符串
            act_limits: 请求限流规则，格式为 [(window, threshold), ...]
            rate_limit: 限流规则，格式为 (requests_per_second, max_bucket_size)
        """
        self._limits = tuple((w, t) for w, t in act_limits if w > 0 and t > 0)
        if not self._limits:
            raise ValueError("至少需要一条有效限流规则")
        self._max_window = max(w for w, _ in self._limits)
        if (
            rate_limit is not None
            and len(rate_limit) == 2
            and rate_limit[0] > 0
            and rate_limit[1] > 0
        ):
            self._use_rate_limit = True
            self._rps, self._cap = rate_limit
        else:
            self._use_rate_limit = False
            self._rps, self._cap = (1.0, 1.0)

        kw = {"poolclass": NullPool} if db_url.startswith("sqlite") else {}
        self._engine: AsyncEngine = create_async_engine(db_url, **kw)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        self._ready = False
        self._lock = asyncio.Lock()

    async def _ensure_schema(self) -> None:
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            async with self._engine.begin() as conn:
                await conn.run_sync(_Base.metadata.create_all)
            self._ready = True

    def _now(self) -> float:
        return datetime.now(tz=UTC).timestamp()

    async def _purge(self, session, sid: str, before: float) -> None:
        await session.execute(
            delete(_Record).where(
                _Record.session_id == sid, _Record.timestamp <= before
            )
        )

    async def _check_bucket(self, session: AsyncSession, sid: str, now: float) -> bool:
        bucket = await session.get(_Bucket, sid)
        if bucket is None:
            if self._cap >= 1.0:
                session.add(
                    _Bucket(
                        session_id=sid,
                        available=self._cap - 1.0,
                        last_update=now,
                    )
                )
                return True
            return False

        elapsed = now - bucket.last_update
        if elapsed > 0:
            bucket.available = min(bucket.available + elapsed * self._rps, self._cap)
            bucket.last_update = now

        if bucket.available >= 1.0:
            bucket.available -= 1.0
            return True
        return False

    async def acquire(
        self,
        session_id: str,
        *,
        timestamp: float | None = None,
    ) -> bool:
        t = timestamp if timestamp is not None else self._now()
        await self._ensure_schema()

        async with self._sessionmaker() as s:
            await self._purge(s, session_id, t - self._max_window)
            for window, threshold in self._limits:
                total = await s.execute(
                    select(func.coalesce(func.sum(_Record.weight), 0.0)).where(
                        _Record.session_id == session_id,
                        _Record.timestamp > t - window,
                    )
                )
                if float(total.scalar_one()) >= threshold:
                    await s.commit()
                    return False
            if self._use_rate_limit and not await self._check_bucket(s, session_id, t):
                await s.commit()
                return False
            await s.commit()
            return True

    async def record(
        self,
        session_id: str,
        *,
        weight: float = 1.0,
        timestamp: float | None = None,
    ) -> None:
        t = timestamp if timestamp is not None else self._now()
        await self._ensure_schema()

        async with self._sessionmaker() as s:
            s.add(_Record(session_id=session_id, timestamp=t, weight=weight))
            await self._purge(s, session_id, t - self._max_window)
            await s.commit()

    async def quota(
        self,
        session_id: str,
        *,
        timestamp: float | None = None,
    ) -> tuple[tuple[int, float], ...]:
        t = timestamp if timestamp is not None else self._now()
        await self._ensure_schema()

        async with self._sessionmaker() as s:
            result: list[tuple[int, float]] = []
            for window, threshold in self._limits:
                total = await s.execute(
                    select(func.coalesce(func.sum(_Record.weight), 0.0)).where(
                        _Record.session_id == session_id,
                        _Record.timestamp > t - window,
                    )
                )
                result.append((window, min(float(total.scalar_one()) / threshold, 1.0)))
            return tuple(result)


class ActivateLimitMiddleware(AgentMiddleware[ManagerState, ManagerContext]):
    def __init__(
        self,
        db_url: str,
        act_limits: tuple[tuple[int, int], ...],
        rate_limit: tuple[float, float] | None = None,
    ) -> None:
        self.limiter = ActivateLimiter(
            db_url=db_url,
            act_limits=act_limits,
            rate_limit=rate_limit,
        )

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        _ = state
        if runtime.context["unrestricted"]:
            return None
        if not await self.limiter.acquire(runtime.context["session_id"]):
            return {
                "jump_to": "end",
                "outputs": [
                    OutputMessage(
                        type="limit",
                        data={
                            "quota": await self.limiter.quota(
                                runtime.context["session_id"]
                            ),
                        },
                    )
                ],
            }
        return None

    async def aafter_agent(
        self, state: ManagerState, runtime: Runtime[ManagerContext]
    ) -> dict[str, Any] | None:
        _ = runtime
        if runtime.context["unrestricted"]:
            return None
        observe = []
        for output in state["outputs"]:
            if output["type"] == "reply":
                observe.append(AIMessage(output["data"]["full_text"]))
            elif output["type"] == "meme":
                observe.append(AIMessage(output["data"]["content"]))
        if observe:
            await self.limiter.record(runtime.context["session_id"])
