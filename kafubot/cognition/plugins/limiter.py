from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict, cast

import anyio
from sqlalchemy import Float, Index, Integer, Text, delete, func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool

from kafubot.cognition.plugins.base import PluginContext, PluginDefinition
from kafubot.cognition.plugins.lifecycle import ReplyCommitted, ReplyPreparation

if TYPE_CHECKING:
    from collections.abc import Sequence


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
    rate_limit: NotRequired[tuple[float, float] | None]
    notice_when_limit: NotRequired[bool]


class ActivateLimiter:
    def __init__(
        self,
        db_url: str,
        act_limits: tuple[tuple[int, int], ...] = (),
        rate_limit: tuple[float, float] | None = None,
    ) -> None:
        """
        Args:
            db_url: 数据库连接字符串
            act_limits: 请求限流规则，格式为 [(window, threshold), ...]
            rate_limit: 限流规则，格式为 (requests_per_second, max_bucket_size)
        """
        self._act_limits = tuple((w, t) for w, t in act_limits if w > 0 and t > 0)
        self._max_window = max(604800, *(w for w, _ in self._act_limits))

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

        kw = (
            {"poolclass": NullPool}
            if db_url.startswith(("sqlite", "aiosqlite"))
            else {}
        )
        self._engine: AsyncEngine = create_async_engine(db_url, **kw)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        self._ready = False
        self._lock = anyio.Lock()

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

    async def _purge(self, session: AsyncSession, sid: str, before: float) -> None:
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
            for window, threshold in self._act_limits:
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

    async def acquire_with_text(
        self,
        session_id: str,
        *,
        timestamp: float | None = None,
    ) -> tuple[bool, str | None]:
        t = timestamp if timestamp is not None else self._now()
        await self._ensure_schema()

        async with self._sessionmaker() as s:
            await self._purge(s, session_id, t - self._max_window)
            result: list[tuple[int, float]] = []
            for window, threshold in self._act_limits:
                total = await s.execute(
                    select(func.coalesce(func.sum(_Record.weight), 0.0)).where(
                        _Record.session_id == session_id,
                        _Record.timestamp > t - window,
                    )
                )
                result.append((window, min(float(total.scalar_one()) / threshold, 1.0)))
            if any(v >= 1.0 for _, v in result):
                await s.commit()
                return False, _quota_to_text(result)
            if self._use_rate_limit and not await self._check_bucket(s, session_id, t):
                await s.commit()
                return False, "发的太快啦~ 让可不休息一会儿吧~"
            await s.commit()
            return True, None

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

    async def aclose(self) -> None:
        await self._engine.dispose()


def _format_duration(seconds: int) -> str:
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    parts = []
    for value, unit in ((d, "day"), (h, "hour"), (m, "min"), (s, "sec")):
        if value:
            parts.append(f"{value}{unit}")
    return " ".join(parts) if parts else "0s"


def _quota_to_text(quota: list[tuple[int, float]]) -> str | None:
    if not quota:
        return None
    texts = ["当前回复额度耗尽，请稍后重试。\n以下是您的额度使用情况：\n"]
    texts.extend(
        [
            f"{int(quota_item[1] * 100)} % was used within {_format_duration(quota_item[0])}"
            for quota_item in quota
        ]
    )
    return "\n".join(texts) or None


def apply(context: PluginContext, config: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "db_url": "sqlite+aiosqlite:///./.database/activate_limiter.db",
        "act_limits": ((18000, 60), (604800, 450)),
        "rate_limit": (0.08, 1.8),
        **config,
    }
    values["act_limits"] = tuple(
        tuple(item) for item in cast("Sequence[Sequence[int]]", values["act_limits"])
    )
    if values.get("rate_limit") is not None:
        values["rate_limit"] = tuple(cast("Sequence[float]", values["rate_limit"]))
    values.pop("notice_when_limit", None)
    limiter = cast("Any", ActivateLimiter)(**values)

    async def guard(preparation: ReplyPreparation) -> str | None:
        allowed, text = await limiter.acquire_with_text(preparation.context.session_id)
        return None if allowed else (text or "reply rate limit reached")

    async def committed(event: ReplyCommitted) -> None:
        await limiter.record(event.preparation.context.session_id)

    context.resource("limiter", limiter)
    context.on_guard_reply(guard)
    context.on_reply_committed(committed)


plugin = PluginDefinition(name="limiter", apply=apply)


__all__ = ["ActivateLimiter", "ActivateLimiterConfig", "plugin"]
