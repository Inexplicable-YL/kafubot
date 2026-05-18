from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from functools import cache
from typing import Literal, cast
from zoneinfo import ZoneInfo

from sqlalchemy import DateTime, Float, Index, Integer, Text, delete, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool

ACTIVITY_DB_URL = os.getenv(
    "ACTIVITY_LIMITS_DB_URL",
    "sqlite+aiosqlite:///./activity_limits.db",
)
ACTIVITY_TABLE_NAME = os.getenv("ACTIVITY_LIMITS_TABLE", "activity_records")


class ActivityBase(DeclarativeBase):
    pass


class ActivityRecord(ActivityBase):
    __tablename__ = ACTIVITY_TABLE_NAME
    __table_args__ = (
        Index(
            f"ix_{ACTIVITY_TABLE_NAME}_scope_session_timestamp",
            "scope",
            "session_id",
            "timestamp",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False
    )
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False
    )


def _utc_naive_from_timestamp(timestamp: int) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=UTC).replace(tzinfo=None)


def _enabled_limits(
    activity_limits: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (window_seconds, threshold)
        for window_seconds, threshold in activity_limits
        if window_seconds > 0 and threshold > 0
    )


class ActivityStore:
    def __init__(self, db_url: str = ACTIVITY_DB_URL) -> None:
        engine_kwargs = {"poolclass": NullPool} if db_url.startswith("sqlite") else {}
        self.engine: AsyncEngine = create_async_engine(db_url, **engine_kwargs)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self.engine.begin() as conn:
                await conn.run_sync(ActivityBase.metadata.create_all)
            self._schema_ready = True

    async def is_limited(
        self,
        *,
        scope: Literal["group", "private"],
        session_id: str,
        event_time: int,
        activity_limits: tuple[tuple[int, int], ...],
    ) -> bool:
        limits = _enabled_limits(activity_limits)
        if not limits:
            return False

        await self._ensure_schema()
        event_datetime = _utc_naive_from_timestamp(event_time)
        max_window = max(window_seconds for window_seconds, _ in limits)

        async with self.sessionmaker() as session:
            await session.execute(
                delete(ActivityRecord).where(
                    ActivityRecord.scope == scope,
                    ActivityRecord.session_id == session_id,
                    ActivityRecord.timestamp
                    <= event_datetime - timedelta(seconds=max_window),
                )
            )

            for window_seconds, threshold in limits:
                total_result = await session.execute(
                    select(func.coalesce(func.sum(ActivityRecord.weight), 0.0)).where(
                        ActivityRecord.scope == scope,
                        ActivityRecord.session_id == session_id,
                        ActivityRecord.timestamp
                        > event_datetime - timedelta(seconds=window_seconds),
                    )
                )
                if float(total_result.scalar_one()) >= threshold:
                    await session.commit()
                    return True

            await session.commit()
            return False

    async def record(
        self,
        *,
        scope: Literal["group", "private"],
        session_id: str,
        event_time: int,
        weight: float,
        activity_limits: tuple[tuple[int, int], ...],
    ) -> None:
        limits = _enabled_limits(activity_limits)
        if not limits:
            return

        await self._ensure_schema()
        event_datetime = _utc_naive_from_timestamp(event_time)
        max_window = max(window_seconds for window_seconds, _ in limits)
        now = datetime.now(UTC).replace(tzinfo=None)

        async with self.sessionmaker() as session:
            session.add(
                ActivityRecord(
                    scope=scope,
                    session_id=session_id,
                    timestamp=event_datetime,
                    weight=weight,
                    created_at=now,
                )
            )
            await session.execute(
                delete(ActivityRecord).where(
                    ActivityRecord.scope == scope,
                    ActivityRecord.session_id == session_id,
                    ActivityRecord.timestamp
                    <= event_datetime - timedelta(seconds=max_window),
                )
            )
            await session.commit()

    async def clear_session(
        self, *, scope: Literal["group", "private"], session_id: str
    ) -> None:
        await self._ensure_schema()
        async with self.sessionmaker() as session:
            await session.execute(
                delete(ActivityRecord).where(
                    ActivityRecord.scope == scope,
                    ActivityRecord.session_id == session_id,
                )
            )
            await session.commit()

    async def latest_timestamp(
        self, *, scope: Literal["group", "private"], session_id: str
    ) -> float | None:
        await self._ensure_schema()
        async with self.sessionmaker() as session:
            result = await session.execute(
                select(ActivityRecord.timestamp)
                .where(
                    ActivityRecord.scope == scope,
                    ActivityRecord.session_id == session_id,
                )
                .order_by(ActivityRecord.timestamp.desc())
                .limit(1)
            )
            row = result.one_or_none()
            if row is None:
                return None
            return cast("datetime", row[0]).replace(tzinfo=ZoneInfo("UTC")).timestamp()

    async def recent_interval(
        self,
        *,
        scope: Literal["group", "private"],
        session_id: str,
        window_seconds: int = 600,
    ) -> list[float] | None:
        await self._ensure_schema()

        async with self.sessionmaker() as session:
            latest_result = await session.execute(
                select(ActivityRecord.timestamp)
                .where(
                    ActivityRecord.scope == scope,
                    ActivityRecord.session_id == session_id,
                )
                .order_by(ActivityRecord.timestamp.desc())
                .limit(1)
            )
            latest_row = latest_result.one_or_none()
            if latest_row is None:
                return None
            latest_timestamp: datetime = latest_row[0]
            window_start = latest_timestamp - timedelta(seconds=window_seconds)

            result = await session.execute(
                select(ActivityRecord.timestamp)
                .where(
                    ActivityRecord.scope == scope,
                    ActivityRecord.session_id == session_id,
                    ActivityRecord.timestamp > window_start,
                    ActivityRecord.timestamp <= latest_timestamp,
                )
                .order_by(ActivityRecord.timestamp)
            )
            timestamps: list[datetime] = [row[0] for row in result.all()]

            if len(timestamps) <= 1:
                return None

            return [
                (timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]


@cache
def get_activity_store() -> ActivityStore:
    return ActivityStore()
