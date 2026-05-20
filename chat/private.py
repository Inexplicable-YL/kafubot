from __future__ import annotations

import os
from datetime import UTC, datetime, tzinfo
from functools import cache
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override
from zoneinfo import ZoneInfo

from langchain_community.chat_message_histories.sql import BaseMessageConverter
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableGenerator,
    RunnableLambda,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, TypeAdapter
from sqlalchemy import DateTime, Integer, Text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from chat.prompt import PRIVATE_SYSTEM_PROMPT
from chat.utils import LimitedSQLChatMessageHistory, content_to_text, to_reply

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


DB_URL = os.getenv(
    "PRIVATE_HISTORY_DB_URL", "sqlite+aiosqlite:///./.database/private_history.db"
)
TABLE_NAME = os.getenv("PRIVATE_HISTORY_TABLE", "deepseek_chat_messages")
DEEPSEEK_MODEL = os.getenv("PRIVATE_DEEPSEEK_MODEL", "deepseek-v4-flash")
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("PRIVATE_CHAT_HISTORY_MAX_MESSAGES", "100"))
MODEL_VISIBLE_TZ = ZoneInfo(os.getenv("MODEL_VISIBLE_TZ", "Asia/Shanghai"))
MODEL_VISIBLE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _format_message_content(
    timestamp: datetime,
    text: str,
    *,
    timezone: tzinfo,
) -> str:
    return f"[{_ensure_utc(timestamp).astimezone(timezone).isoformat()}]{text}"


def _format_model_visible_message_content(
    timestamp: datetime,
    text: str,
    *,
    timezone: tzinfo,
) -> str:
    visible_timestamp = _ensure_utc(timestamp).astimezone(timezone)
    return f"[{visible_timestamp.strftime(MODEL_VISIBLE_TIME_FORMAT)}]{text}"


@cache
def _get_async_engine() -> AsyncEngine:
    return create_async_engine(DB_URL)


class UserMessage(BaseModel):
    timestamp: datetime
    text: str

    def as_content(self, *, timezone: tzinfo = MODEL_VISIBLE_TZ) -> str:
        return _format_model_visible_message_content(
            self.timestamp,
            self.text,
            timezone=timezone,
        )

    def as_human_message(self, *, timezone: tzinfo = MODEL_VISIBLE_TZ) -> HumanMessage:
        return HumanMessage(
            content=self.as_content(timezone=timezone),
            additional_kwargs={
                "raw": {
                    "timestamp": _ensure_utc(self.timestamp),
                    "text": self.text,
                }
            },
        )


class ChatMessageBase(DeclarativeBase):
    pass


class ChatMessageRecord(ChatMessageBase):
    __tablename__ = TABLE_NAME

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    user_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    user_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    assistant_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class MessageConverter(BaseMessageConverter):
    def get_sql_model_class(self) -> Any:
        return ChatMessageRecord

    @override
    def from_sql_model(self, sql_message: ChatMessageRecord) -> BaseMessage:
        role = sql_message.role
        if role == "human":
            user_timestamp = sql_message.user_timestamp
            user_text = sql_message.user_text
            if user_timestamp is not None and user_text is not None:
                return HumanMessage(
                    content=_format_model_visible_message_content(
                        user_timestamp,
                        user_text,
                        timezone=MODEL_VISIBLE_TZ,
                    ),
                    additional_kwargs={
                        "raw": {
                            "timestamp": _ensure_utc(user_timestamp),
                            "text": user_text,
                        }
                    },
                )
            return HumanMessage(content=sql_message.content)
        if role == "ai":
            return AIMessage(content=sql_message.content)
        raise ValueError(f"Unknown message role: {role}")

    @override
    def to_sql_model(self, message: BaseMessage, session_id: str) -> ChatMessageRecord:
        now = datetime.now(UTC)

        if isinstance(message, HumanMessage):
            raw = message.additional_kwargs.get("raw", {})
            raw_timestamp = raw.get("timestamp")
            raw_text = raw.get("text")
            user_timestamp = (
                _ensure_utc(raw_timestamp)
                if isinstance(raw_timestamp, datetime)
                else None
            )
            user_text = raw_text if isinstance(raw_text, str) else None
            content = (
                _format_message_content(
                    user_timestamp,
                    user_text,
                    timezone=UTC,
                )
                if user_timestamp is not None and user_text is not None
                else content_to_text(message.content)
            )
            return ChatMessageRecord(
                session_id=session_id,
                role="human",
                content=content,
                created_at=now,
                user_timestamp=user_timestamp,
                user_text=user_text,
                assistant_timestamp=None,
            )

        if isinstance(message, AIMessage):
            assistant_timestamp = message.additional_kwargs.get(
                "assistant_timestamp",
                now,
            )
            if not isinstance(assistant_timestamp, datetime):
                assistant_timestamp = now
            return ChatMessageRecord(
                session_id=session_id,
                role="ai",
                content=content_to_text(message.content),
                created_at=now,
                user_timestamp=None,
                user_text=None,
                assistant_timestamp=_ensure_utc(assistant_timestamp),
            )

        raise TypeError(f"Unsupported message type: {type(message)}")


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
        messages = TypeAdapter(list[UserMessage]).validate_python(payload["messages"])
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
            "current_messages": [message.as_human_message() for message in messages],
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
    model = ChatDeepSeek(
        model=DEEPSEEK_MODEL,
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        temperature=1.2,
        max_retries=2,
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

        return cast(
            "Runnable[dict[str, Any], AIMessage]", prompt | model.bind(**runtime_kwargs)
        )

    async def _attach_timestamp(
        messages: AsyncIterator[AIMessage],
    ) -> AsyncIterator[AIMessage]:
        timestamp = datetime.now(UTC)
        async for message in messages:
            extra = dict(message.additional_kwargs or {})
            extra["assistant_timestamp"] = timestamp
            yield message.model_copy(update={"additional_kwargs": extra})

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
        | RunnableGenerator(to_reply)
    )
