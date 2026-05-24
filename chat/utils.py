from __future__ import annotations

import re
from collections import OrderedDict
from math import exp, sqrt, tanh
from typing import TYPE_CHECKING, Any, ClassVar

import anyio
from langchain_community.chat_message_histories.sql import (
    SQLChatMessageHistory,
)
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from sqlalchemy import delete, select

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from langchain_core.messages import AIMessage, BaseMessage


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item
            if isinstance(item, str)
            else str(item.get("text") or item.get("content") or item)
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    return str(content)


class LimitedSQLChatMessageHistory(SQLChatMessageHistory):
    _messages_cache: ClassVar[OrderedDict[str, list[BaseMessage]]] = OrderedDict()
    _session_locks: ClassVar[dict[str, anyio.Lock]] = {}
    _locks_lock: ClassVar[anyio.Lock] = anyio.Lock()
    _MAX_CACHE_SIZE: ClassVar[int] = 256

    @classmethod
    async def _get_session_lock(cls, session_id: str) -> anyio.Lock:
        lock = cls._session_locks.get(session_id)
        if lock is not None:
            return lock
        async with cls._locks_lock:
            if session_id not in cls._session_locks:
                cls._session_locks[session_id] = anyio.Lock()
            return cls._session_locks[session_id]

    def __init__(
        self,
        *args: Any,
        max_messages: int | None = None,
        **kwargs: Any,
    ) -> None:
        if max_messages is not None and max_messages < 1:
            raise ValueError("max_messages must be positive or None")
        self.max_messages = max_messages
        super().__init__(*args, **kwargs)

    async def aget_messages(self) -> list[BaseMessage]:
        lock = await self._get_session_lock(self.session_id)
        async with lock:
            cached = self._messages_cache.get(self.session_id)
            if cached is not None:
                self._messages_cache.move_to_end(self.session_id)
                return list(cached)

            if self.max_messages is None:
                messages = await super().aget_messages()
            else:
                await self._acreate_table_if_not_exists()
                session_id_field = getattr(
                    self.sql_model_class, self.session_id_field_name
                )

                async with self._make_async_session() as session:
                    stmt = (
                        select(self.sql_model_class)
                        .where(session_id_field == self.session_id)
                        .order_by(self.sql_model_class.id.desc())
                        .limit(self.max_messages)
                    )
                    result = await session.execute(stmt)
                    records = list(result.scalars())

                messages = [
                    self.converter.from_sql_model(record)
                    for record in reversed(records)
                ]

            if (
                len(self._messages_cache) >= self._MAX_CACHE_SIZE
                and self.session_id not in self._messages_cache
            ):
                self._messages_cache.popitem(last=False)

            self._messages_cache[self.session_id] = messages
            return list(messages)

    async def aadd_message(self, message: BaseMessage) -> None:
        lock = await self._get_session_lock(self.session_id)
        async with lock:
            cached = self._messages_cache.get(self.session_id)
            await super().aadd_message(message)

            # 若缓存不存在或已达上限才执行 prune，减少无意义查询
            if self.max_messages is not None and (
                cached is None or len(cached) >= self.max_messages
            ):
                await self._aprune_messages()

            if cached is not None:
                new_cache = cached + [message]
                if self.max_messages is not None and len(new_cache) > self.max_messages:
                    new_cache = new_cache[-self.max_messages :]
                self._messages_cache[self.session_id] = new_cache

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        lock = await self._get_session_lock(self.session_id)
        async with lock:
            cached = self._messages_cache.get(self.session_id)
            await super().aadd_messages(messages)

            if self.max_messages is not None and (
                cached is None or len(cached) + len(messages) > self.max_messages
            ):
                await self._aprune_messages()

            if cached is not None:
                new_cache = list(cached)
                new_cache.extend(messages)
                if self.max_messages is not None and len(new_cache) > self.max_messages:
                    new_cache = new_cache[-self.max_messages :]
                self._messages_cache[self.session_id] = new_cache

    async def aclear(self) -> None:
        lock = await self._get_session_lock(self.session_id)
        async with lock:
            await super().aclear()
            self._messages_cache.pop(self.session_id, None)

    async def _aprune_messages(self) -> None:
        if self.max_messages is None:
            return

        await self._acreate_table_if_not_exists()
        session_id_field = getattr(self.sql_model_class, self.session_id_field_name)
        id_field = self.sql_model_class.id

        async with self._make_async_session() as session:
            ids_result = await session.execute(
                select(id_field)
                .where(session_id_field == self.session_id)
                .order_by(id_field.desc())
                .offset(self.max_messages)
            )
            ids_to_delete = list(ids_result.scalars())
            if not ids_to_delete:
                return

            await session.execute(
                delete(self.sql_model_class).where(id_field.in_(ids_to_delete))
            )
            await session.commit()


class Segment(BaseModel):
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


def parse_message(text: str) -> tuple[list[Segment], str]:
    text = text.replace("：", ":").strip()
    segments = []
    pattern = re.compile(r"\[MSG:[^\]]*\]")

    def replacer(match: re.Match) -> str:
        block = match.group(0)
        inner = block[5:-1]
        if "," in inner:
            type_part, params_str = inner.split(",", 1)
        else:
            type_part = inner
            params_str = ""
        typ = type_part.strip()
        if not typ:
            seg = None
        data = {}
        if params_str:
            params_with_end = params_str.strip() + ","
            pairs = re.findall(r"([^\s,=]+)\s*=\s*([^,]*?)\s*(?=,)", params_with_end)
            data = {k.strip(): v.strip() for k, v in pairs}
        seg = Segment(type=typ, data=data)
        if seg:
            segments.append(seg)
            return ""
        return block

    clean_text = pattern.sub(replacer, text)
    return segments, clean_text.strip()


async def to_reply(
    messages: AsyncIterator[AIMessage],
) -> AsyncIterator[str]:
    def _process_message_chunk(
        text: str,
        answer: str,
    ) -> tuple[str, str | None]:
        text = text.strip(" ")
        reply: str | None = None
        if not text:
            return answer, reply

        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i < len(lines) - 1:
                answer += stripped
                if answer:
                    reply = answer
                answer = ""
            else:
                answer += stripped
        return answer, reply

    answer = ""
    async for message in messages:
        answer, reply = _process_message_chunk(
            content_to_text(message.content),
            answer,
        )
        if reply is not None:
            yield reply + ("\n" if not reply.endswith("\n") else "")
    if answer.strip():
        yield answer.strip()


def terminal_trend(
    xs: list[bool],
    slow: float = 2.0,
    fast: float = 3.0,
    scale: float = 2.7,
    damp: float = 0.7,
) -> float:
    n: int = len(xs)
    if n < 8:  # noqa: PLR2004
        return 0.0
    true_count: int = sum(1 for x in xs if x)
    if true_count in (0, n):
        return 0.0
    p: float = true_count / n
    sd: float = sqrt(p * (1.0 - p))
    if sd < 1e-12:  # noqa: PLR2004
        return 0.0
    ys: list[float] = [1.0 if x else 0.0 for x in xs]
    zs: list[float] = [(y - p) / sd for y in ys]
    ts: list[float] = [i / (n - 1) for i in range(n)]
    slopes: list[float] = []
    for lam in (slow, fast):
        weights: list[float] = [exp(lam * (t - 1.0)) for t in ts]
        weight_sum: float = sum(weights)
        t_mean: float = (
            sum(w * t for w, t in zip(weights, ts, strict=False)) / weight_sum
        )
        z_mean: float = (
            sum(w * z for w, z in zip(weights, zs, strict=False)) / weight_sum
        )
        covariance: float = sum(
            w * (t - t_mean) * (z - z_mean)
            for w, t, z in zip(weights, ts, zs, strict=False)
        )
        variance: float = sum(
            w * (t - t_mean) ** 2 for w, t in zip(weights, ts, strict=False)
        )
        slope: float = 0.0 if variance < 1e-12 else covariance / variance  # noqa: PLR2004
        slopes.append(slope)
    slow_slope: float = slopes[0]
    fast_slope: float = slopes[1]
    raw: float
    if slow_slope * fast_slope < 0:
        raw = 0.5 * fast_slope
    elif abs(fast_slope) < abs(slow_slope):
        flatten_ratio: float = (abs(slow_slope) - abs(fast_slope)) / (
            abs(slow_slope) + 1e-12
        )
        raw = fast_slope * (1.0 - damp * flatten_ratio)
    else:
        raw = fast_slope
    return tanh(raw / scale)
