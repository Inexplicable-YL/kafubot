from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import UTC, datetime, tzinfo
from functools import cache
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override
from zoneinfo import ZoneInfo

import anyio
from _prompt import DECISION_SYSTEM_PROMPT, GROUP_SYSTEM_PROMPT
from langchain_community.chat_message_histories.sql import (
    BaseMessageConverter,
    SQLChatMessageHistory,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableGenerator,
    RunnableLambda,
)
from langchain_core.runnables.branch import RunnableBranch
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, TypeAdapter
from sqlalchemy import Column, DateTime, Integer, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import declarative_base

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from anyio.streams.memory import MemoryObjectSendStream


DB_URL = os.getenv("CHAT_HISTORY_DB_URL", "sqlite+aiosqlite:///./group_history.db")
TABLE_NAME = os.getenv("CHAT_HISTORY_TABLE", "deepseek_chat_messages")

DECISION_DEEPSEEK_MODEL = os.getenv("DECISION_DEEPSEEK_MODEL", "deepseek-v4-flash")
CHAT_DEEPSEEK_MODEL = os.getenv("CHAT_DEEPSEEK_MODEL", "deepseek-v4-pro")

MODEL_VISIBLE_TZ = ZoneInfo(os.getenv("MODEL_VISIBLE_TZ", "Asia/Shanghai"))
MODEL_VISIBLE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "50"))


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _format_message_content(
    timestamp: datetime,
    user: str,
    text: str,
    *,
    timezone: tzinfo,
) -> str:
    return f"[{_ensure_utc(timestamp).astimezone(timezone).isoformat()}]{user}: {text}"


def _format_model_visible_message_content(
    timestamp: datetime,
    user: str,
    text: str,
    *,
    timezone: tzinfo,
) -> str:
    visible_timestamp = _ensure_utc(timestamp).astimezone(timezone)
    return f"[{visible_timestamp.strftime(MODEL_VISIBLE_TIME_FORMAT)}]{user}: {text}"


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


class UserMessage(BaseModel):
    timestamp: datetime
    user: str
    text: str

    def as_content(self, *, timezone: tzinfo = MODEL_VISIBLE_TZ) -> str:
        return _format_model_visible_message_content(
            self.timestamp,
            self.user,
            self.text,
            timezone=timezone,
        )


class AssistantReply(BaseModel):
    timestamp: datetime
    text: str

    def __add__(self, other: AssistantReply) -> AssistantReply:
        return AssistantReply(timestamp=self.timestamp, text=self.text + other.text)


class ChatMessageRecord(declarative_base()):
    __tablename__ = TABLE_NAME

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Text, index=True, nullable=False)
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)
    user_timestamp = Column(DateTime(timezone=True), nullable=True)
    user_name = Column(Text, nullable=True)
    user_text = Column(Text, nullable=True)
    assistant_timestamp = Column(DateTime(timezone=True), nullable=True)


class MessageConverter(BaseMessageConverter):
    def get_sql_model_class(self) -> Any:
        return ChatMessageRecord

    @override
    def from_sql_model(self, sql_message: ChatMessageRecord) -> BaseMessage:
        role = cast("str", sql_message.role)
        if role == "human":
            user_timestamp = cast("datetime | None", sql_message.user_timestamp)
            user_name = cast("str | None", sql_message.user_name)
            user_text = cast("str | None", sql_message.user_text)
            if (
                user_timestamp is not None
                and user_name is not None
                and user_text is not None
            ):
                return HumanMessage(
                    content=_format_model_visible_message_content(
                        user_timestamp,
                        user_name,
                        user_text,
                        timezone=MODEL_VISIBLE_TZ,
                    ),
                    additional_kwargs={
                        "raw": {
                            "timestamp": _ensure_utc(user_timestamp),
                            "user": user_name,
                            "text": user_text,
                        }
                    },
                )
            return HumanMessage(content=cast("str", sql_message.content))
        if role == "ai":
            return AIMessage(content=cast("str", sql_message.content))
        raise ValueError(f"Unknown message role: {role}")

    @override
    def to_sql_model(self, message: BaseMessage, session_id: str) -> ChatMessageRecord:
        now = datetime.now(UTC)

        if isinstance(message, HumanMessage):
            raw = message.additional_kwargs.get("raw", {})
            raw_timestamp = raw.get("timestamp")
            raw_user = raw.get("user")
            raw_text = raw.get("text")
            content = (
                _format_message_content(
                    raw_timestamp,
                    raw_user,
                    raw_text,
                    timezone=UTC,
                )
                if isinstance(raw_timestamp, datetime)
                and isinstance(raw_user, str)
                and isinstance(raw_text, str)
                else content_to_text(message.content)
            )
            return ChatMessageRecord(
                session_id=session_id,
                role="human",
                content=content,
                created_at=now,
                user_timestamp=_ensure_utc(raw_timestamp)
                if isinstance(raw_timestamp, datetime)
                else None,
                user_name=raw_user,
                user_text=raw_text,
                assistant_timestamp=None,
            )

        if isinstance(message, AIMessage):
            return ChatMessageRecord(
                session_id=session_id,
                role="ai",
                content=content_to_text(message.content),
                created_at=now,
                user_timestamp=None,
                user_name=None,
                user_text=None,
                assistant_timestamp=message.additional_kwargs.get(
                    "assistant_timestamp",
                    now,
                ),
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


def get_decision_app() -> Runnable[dict[str, Any], bool]:
    def _normalize_input(payload: dict[str, Any]) -> dict[str, Any]:
        messages = TypeAdapter(list[UserMessage]).validate_python(payload["messages"])
        prompt_variables = {
            key: value
            for key, value in payload.items()
            if key not in {"messages", "thinking", "reasoning_effort"}
        }
        return {
            **prompt_variables,
            "current_messages": [
                HumanMessage(
                    content=item.as_content(timezone=MODEL_VISIBLE_TZ),
                    additional_kwargs={
                        "raw": {
                            "timestamp": _ensure_utc(item.timestamp),
                            "user": item.user,
                            "text": item.text,
                        }
                    },
                )
                for item in messages
            ],
        }

    async def _load_history(
        payload: dict[str, Any],
        config: RunnableConfig,
    ) -> dict[str, Any]:
        configurable = config.get("configurable", {})
        session_id = configurable.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("config['configurable']['session_id'] is required")

        history = await get_session_history(session_id).aget_messages()
        history = [
            HumanMessage(content="可不（机器人）: " + content_to_text(message.content))
            if isinstance(message, AIMessage)
            else message
            for message in history
        ]
        return {
            **payload,
            "decision_messages": [
                *history,
                *payload["current_messages"],
            ],
        }

    def _parse_decision(result: Any) -> bool:
        text = content_to_text(
            result.content if isinstance(result, AIMessage) else result
        )
        normalized = text.strip().lower().strip("`'\". \t\r\n")
        if normalized in {"true", "yes", "1"} or normalized.startswith("true"):
            return True
        if normalized in {"false", "no", "0"} or normalized.startswith("false"):
            return False
        raise ValueError(f"Decision model returned an invalid boolean: {text!r}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DECISION_SYSTEM_PROMPT),
            MessagesPlaceholder("decision_messages"),
        ]
    )
    model = ChatDeepSeek(
        model=DECISION_DEEPSEEK_MODEL,
        temperature=0,
        max_retries=2,
    ).bind(
        reasoning_effort="max",
        extra_body={
            "thinking": {
                "type": "enabled",
            }
        },
    )

    return (
        RunnableLambda(_normalize_input)
        | RunnableLambda(_load_history)
        | prompt
        | model
        | RunnableLambda(_parse_decision)
    )


def get_chat_app() -> Runnable[dict[str, Any], AssistantReply]:  # noqa: PLR0915
    def _normalize_input(payload: dict[str, Any]) -> dict[str, Any]:
        messages = TypeAdapter(list[UserMessage]).validate_python(payload["messages"])
        reasoning_effort = payload.get("reasoning_effort", "high")
        prompt_variables = {
            key: value
            for key, value in payload.items()
            if key not in {"messages", "thinking", "reasoning_effort"}
        }

        if reasoning_effort not in {"high", "max"}:
            raise ValueError("reasoning_effort must be 'high' or 'max'")

        return {
            **prompt_variables,
            "current_messages": [
                HumanMessage(
                    content=item.as_content(timezone=MODEL_VISIBLE_TZ),
                    additional_kwargs={
                        "raw": {
                            "timestamp": _ensure_utc(item.timestamp),
                            "user": item.user,
                            "text": item.text,
                        }
                    },
                )
                for item in messages
            ],
            "thinking": bool(payload.get("thinking", False)),
            "reasoning_effort": reasoning_effort,
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GROUP_SYSTEM_PROMPT),
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
            model=CHAT_DEEPSEEK_MODEL,
            temperature=1.2,
            max_retries=2,
        ).bind(**runtime_kwargs)
        return cast("Runnable[dict[str, Any], AIMessage]", prompt | model)

    async def _attach_timestamp(
        messages: AsyncIterator[AIMessage],
    ) -> AsyncIterator[AIMessage]:
        timestamp = datetime.now(UTC)
        async for message in messages:
            extra = dict(message.additional_kwargs or {})
            extra["assistant_timestamp"] = timestamp
            yield message.model_copy(update={"additional_kwargs": extra})

    async def _to_reply(
        messages: AsyncIterator[AIMessage],
    ) -> AsyncIterator[AssistantReply]:
        input_stream, output_stream = anyio.create_memory_object_stream[AssistantReply](
            max_buffer_size=10
        )

        async def trans_and_send(
            msgs: AsyncIterator[AIMessage],
            stream: MemoryObjectSendStream[AssistantReply],
        ) -> None:
            async def _process_message_chunk(
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
            async for message in msgs:
                answer, reply = await _process_message_chunk(
                    content_to_text(message.content),
                    answer,
                )
                if reply is not None:
                    chunk = AssistantReply(
                        timestamp=message.additional_kwargs.get(
                            "assistant_timestamp",
                            datetime.now(UTC),
                        ),
                        text=reply + ("\n" if not reply.endswith("\n") else ""),
                    )
                    await stream.send(chunk)
            if answer.strip():
                chunk = AssistantReply(
                    timestamp=datetime.now(UTC),
                    text=answer.strip(),
                )
                await stream.send(chunk)
            await stream.aclose()

        task = asyncio.create_task(trans_and_send(messages, input_stream))
        try:
            async for reply in output_stream:
                yield reply
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    core_chain = RunnableLambda(_chat_chain_for_payload) | RunnableGenerator(
        _attach_timestamp
    )

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


def get_agent_app() -> Runnable[dict[str, Any], None | AssistantReply]:
    decision_app = get_decision_app()
    chat_app = get_chat_app()

    def is_tome(_input: dict[str, Any]) -> bool:
        return _input.get("is_tome", False)

    return RunnableBranch(
        (is_tome, chat_app),
        (decision_app, chat_app),
        lambda _: None,
    )


if __name__ == "__main__":

    async def main() -> None:
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise RuntimeError("DEEPSEEK_API_KEY is required")

        agent_app = get_agent_app()

        session_id = "group-chat-001"

        messages1 = [
            {
                "timestamp": "2026-05-10T08:00:00",
                "user": "张三",
                "text": "又是早八。",
            },
            {
                "timestamp": "2026-05-10T08:03:08",
                "user": "李四",
                "text": "好累啊，不想去上班了。",
            },
            {
                "timestamp": "2026-05-10T08:07:15",
                "user": "王五",
                "text": "早上连饭都没来得及吃。",
            },
        ]
        messages2 = [
            {
                "timestamp": "2026-05-10T08:10:00",
                "user": "张三",
                "text": "如果没有可不的歌，真的活不下去了。还好有歌听。",
            }
        ]

        reply1: AssistantReply | None = None
        async for reply in agent_app.astream(
            {
                "messages": messages1,
                "now_time": "2026-05-10 08:08:00",
                "thinking": True,
                "reasoning_effort": "high",
                "is_tome": True,
            },
            config={"configurable": {"session_id": session_id}},
        ):
            if reply is not None:
                print(reply.text, end="")
                reply1 = reply if reply1 is None else reply1 + reply
        if reply1 is not None:
            messages1 = []
            print("First round reply:", reply1.model_dump_json(indent=2))

        reply2: AssistantReply | None = None
        async for reply in agent_app.astream(
            {
                "messages": messages1 + messages2,
                "now_time": "2026-05-10 08:11:00",
                "thinking": True,
                "reasoning_effort": "high",
                "is_tome": False,
            },
            config={"configurable": {"session_id": session_id}},
        ):
            if reply is not None:
                print(reply.text, end="")
                reply2 = reply if reply2 is None else reply2 + reply
        if reply2 is not None:
            print("Second round reply:", reply2.model_dump_json(indent=2))

    asyncio.run(main())
