from __future__ import annotations

import os
from functools import cache
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

from langchain_community.chat_message_histories.sql import (
    BaseMessageConverter,
    SQLChatMessageHistory,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableGenerator,
    RunnableLambda,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_deepseek import ChatDeepSeek
from pydantic import TypeAdapter
from sqlalchemy import Integer, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from chat.prompt import PRIVATE_SYSTEM_PROMPT

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence


DB_URL = os.getenv("PRIVATE_HISTORY_DB_URL", "sqlite+aiosqlite:///./private_history.db")
TABLE_NAME = os.getenv("PRIVATE_HISTORY_TABLE", "deepseek_chat_messages")
DEEPSEEK_MODEL = os.getenv("PRIVATE_DEEPSEEK_MODEL", "deepseek-v4-flash")
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("PRIVATE_CHAT_HISTORY_MAX_MESSAGES", "500"))


@cache
def _get_async_engine() -> AsyncEngine:
    return create_async_engine(DB_URL)


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


class ChatMessageBase(DeclarativeBase):
    pass


class ChatMessageRecord(ChatMessageBase):
    __tablename__ = TABLE_NAME

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)


class MessageConverter(BaseMessageConverter):
    def get_sql_model_class(self) -> Any:
        return ChatMessageRecord

    @override
    def from_sql_model(self, sql_message: ChatMessageRecord) -> BaseMessage:
        role = sql_message.role
        if role == "human":
            return HumanMessage(content=sql_message.content)
        if role == "ai":
            return AIMessage(content=sql_message.content)
        raise ValueError(f"Unknown message role: {role}")

    @override
    def to_sql_model(self, message: BaseMessage, session_id: str) -> ChatMessageRecord:
        if isinstance(message, HumanMessage):
            return ChatMessageRecord(
                session_id=session_id,
                role="human",
                content=content_to_text(message.content),
            )

        if isinstance(message, AIMessage):
            return ChatMessageRecord(
                session_id=session_id,
                role="ai",
                content=content_to_text(message.content),
            )

        raise TypeError(f"Unsupported message type: {type(message)}")


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


def get_session_history(session_id: str) -> LimitedSQLChatMessageHistory:
    return LimitedSQLChatMessageHistory(
        session_id=session_id,
        connection=_get_async_engine(),
        custom_message_converter=MessageConverter(),
        max_messages=CHAT_HISTORY_MAX_MESSAGES or None,
    )


async def clear_session_history(session_id: str) -> None:
    await get_session_history(session_id).aclear()


def get_chat_app() -> Runnable[dict[str, Any], str]:
    def _normalize_input(payload: dict[str, Any]) -> dict[str, Any]:
        messages = TypeAdapter(list[str]).validate_python(payload["messages"])
        reasoning_effort = payload.get("reasoning_effort", "high")
        prompt_variables = {
            key: value
            for key, value in payload.items()
            if key not in {"messages", "thinking", "reasoning_effort"}
        }
        prompt_variables.setdefault("extra_prompt", "")
        prompt_variables.setdefault("user_name", "")

        if reasoning_effort not in {"high", "max"}:
            raise ValueError("reasoning_effort must be 'high' or 'max'")

        return {
            **prompt_variables,
            "current_messages": [HumanMessage(content=message) for message in messages],
            "thinking": bool(payload.get("thinking", False)),
            "reasoning_effort": reasoning_effort,
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PRIVATE_SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            MessagesPlaceholder("current_messages"),
        ]
    )

    def _chat_chain_for_payload(
        payload: dict[str, Any],
    ) -> Runnable[dict[str, Any], AIMessage]:
        if bool(payload["thinking"]):
            runtime_kwargs = {
                "reasoning_effort": payload["reasoning_effort"]
                if payload["reasoning_effort"] == "max"
                else "high",
                "extra_body": {
                    "thinking": {
                        "type": "enabled",
                    }
                },
            }
        else:
            runtime_kwargs = {
                "extra_body": {
                    "thinking": {
                        "type": "disabled",
                    }
                },
            }

        model = ChatDeepSeek(
            model=DEEPSEEK_MODEL,
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
            temperature=1.2,
            max_retries=2,
        ).bind(**runtime_kwargs)
        return cast("Runnable[dict[str, Any], AIMessage]", prompt | model)

    async def _to_reply(
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

    core_chain = RunnableLambda(_chat_chain_for_payload)

    chain_with_history = RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="current_messages",
        history_messages_key="history",
    )

    return (
        RunnableLambda(_normalize_input)
        | chain_with_history
        | RunnableGenerator(_to_reply)
    )
