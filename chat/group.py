import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, tzinfo
from functools import cache
from typing import Any, Literal, cast
from typing_extensions import override
from zoneinfo import ZoneInfo

from langchain_community.chat_message_histories.sql import BaseMessageConverter
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableConfig,
    RunnableGenerator,
    RunnableLambda,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from sqlalchemy import DateTime, Integer, Text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from chat.image import ImageReadResult
from chat.message import QQMessage
from chat.prompt import DECISION_SYSTEM_PROMPT, GROUP_SYSTEM_PROMPT
from chat.utils import LimitedSQLChatMessageHistory, content_to_text, to_reply

DB_URL = os.getenv(
    "CHAT_HISTORY_DB_URL", "sqlite+aiosqlite:///./.database/group_history.db"
)
TABLE_NAME = os.getenv("CHAT_HISTORY_TABLE", "deepseek_chat_messages")

DECISION_DEEPSEEK_MODEL = os.getenv("DECISION_DEEPSEEK_MODEL", "deepseek-v4-flash")
CHAT_DEEPSEEK_MODEL = os.getenv("CHAT_DEEPSEEK_MODEL", "deepseek-v4-pro")

MODEL_VISIBLE_TZ = ZoneInfo(os.getenv("MODEL_VISIBLE_TZ", "Asia/Shanghai"))
MODEL_VISIBLE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CHAT_HISTORY_MAX_MESSAGES = int(os.getenv("CHAT_HISTORY_MAX_MESSAGES", "50"))


def _format_message_content(
    timestamp: datetime,
    user: str,
    text: str,
    *,
    timezone: tzinfo,
) -> str:
    return f"[{timestamp.astimezone(timezone).strftime('%Y-%m-%d %H:%M:%S')}]{user}: {text}"


def _format_model_visible_message_content(
    timestamp: datetime,
    user: str,
    text: str,
    *,
    timezone: tzinfo,
) -> str:
    visible_timestamp = timestamp.astimezone(timezone)
    return f"[{visible_timestamp.strftime(MODEL_VISIBLE_TIME_FORMAT)}]{user}: {text}"


@cache
def _get_async_engine() -> AsyncEngine:
    return create_async_engine(DB_URL)


class GroupMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["user", "assistant"] = "user"
    timestamp: datetime
    user: str
    message: QQMessage
    user_id: str
    message_id: str
    is_tome: bool
    to_other: bool
    have_keywords: bool
    images: list[tuple[ImageReadResult, bool]] = Field(default_factory=list)

    def as_content(self, *, timezone: tzinfo = MODEL_VISIBLE_TZ) -> str:
        return _format_model_visible_message_content(
            self.timestamp,
            self.user,
            self.message.get_msgcode(),
            timezone=timezone,
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
    user_name: Mapped[str | None] = mapped_column(Text, nullable=True)
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
            user_name = sql_message.user_name
            user_text = sql_message.user_text
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
                        "timestamp": user_timestamp,
                        "user": user_name,
                        "text": user_text,
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
            timestamp = message.additional_kwargs.get("timestamp")
            user = message.additional_kwargs.get("user")
            text = message.additional_kwargs.get("text")
            content = (
                _format_message_content(
                    timestamp,
                    user,
                    text,
                    timezone=UTC,
                )
                if isinstance(timestamp, datetime)
                and isinstance(user, str)
                and isinstance(text, str)
                else content_to_text(message.content)
            )
            return ChatMessageRecord(
                session_id=session_id,
                role="human",
                content=content,
                created_at=now,
                user_timestamp=timestamp if isinstance(timestamp, datetime) else None,
                user_name=user,
                user_text=text,
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
                    "timestamp",
                    now,
                ),
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


def get_decision_app() -> Runnable[dict[str, Any], bool]:
    def _normalize_input(payload: dict[str, Any]) -> dict[str, Any]:
        messages = TypeAdapter(list[GroupMessage]).validate_python(payload["messages"])
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
                        "timestamp": item.timestamp,
                        "user": item.user,
                        "text": item.message.get_msgcode(),
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
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
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


def get_chat_app() -> Runnable[dict[str, Any], str]:  # noqa: PLR0915
    def _normalize_input(payload: dict[str, Any]) -> dict[str, Any]:
        messages = TypeAdapter(list[GroupMessage]).validate_python(payload["messages"])
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
                        "timestamp": item.timestamp,
                        "user": item.user,
                        "text": item.message.get_msgcode(),
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
            base_url=os.getenv("DEEPSEEK_BASE_URL"),
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
            extra["timestamp"] = timestamp
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
