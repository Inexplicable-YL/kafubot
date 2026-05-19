from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

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
        if self.max_messages is None:
            return await super().aget_messages()

        await self._acreate_table_if_not_exists()
        session_id_field = getattr(self.sql_model_class, self.session_id_field_name)

        async with self._make_async_session() as session:
            stmt = (
                select(self.sql_model_class)
                .where(session_id_field == self.session_id)
                .order_by(self.sql_model_class.id.desc())
                .limit(self.max_messages)
            )
            result = await session.execute(stmt)
            records = list(result.scalars())

        return [self.converter.from_sql_model(record) for record in reversed(records)]

    async def aadd_message(self, message: BaseMessage) -> None:
        await super().aadd_message(message)
        await self._aprune_messages()

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        await super().aadd_messages(messages)
        await self._aprune_messages()

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
