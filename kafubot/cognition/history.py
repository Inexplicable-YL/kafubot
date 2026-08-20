import json
import re
from datetime import UTC, datetime
from functools import cache
from html import unescape
from typing import Any, Literal
from typing_extensions import override

import imagehash
from langchain_community.chat_message_histories.sql import BaseMessageConverter
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import Boolean, DateTime, Integer, Text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from kafubot.cognition.media.image import ImageReadResult
from kafubot.cognition.message import QQMessage
from kafubot.cognition.types import UserMessage
from kafubot.cognition.utils import LimitedSQLChatMessageHistory, content_to_text

DB_URL = "sqlite+aiosqlite:///./.database/agent_history.db"
TABLE_NAME = "agent_history"

MODEL_VISIBLE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CHAT_HISTORY_MAX_MESSAGES = 100
_QQ_METADATA_RE = re.compile(r"<qq-user-message\s+metadata=(['\"])(.*?)\1>", re.DOTALL)


def _extract_qq_metadata(content: str) -> dict[str, Any]:
    match = _QQ_METADATA_RE.search(content)
    if not match:
        return {}
    try:
        payload = json.loads(unescape(match.group(2)))
    except (json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


@cache
def _get_async_engine() -> AsyncEngine:
    return create_async_engine(DB_URL)


class ChatMessageBase(DeclarativeBase):
    pass


class ChatMessageRecord(ChatMessageBase):
    __tablename__ = TABLE_NAME

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(Text, index=True, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    user: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[str | None] = mapped_column(Text)
    message_id: Mapped[str | None] = mapped_column(Text)
    is_tome: Mapped[bool] = mapped_column(Boolean, default=False)
    message: Mapped[str | None] = mapped_column(Text)
    images: Mapped[str | None] = mapped_column(Text)


class MessageConverter(BaseMessageConverter):
    @override
    def get_sql_model_class(self) -> Any:
        return ChatMessageRecord

    @override
    def from_sql_model(self, sql_message: ChatMessageRecord) -> BaseMessage:
        role = sql_message.role
        if role == "human":
            if sql_message.timestamp is not None and sql_message.message is not None:
                metadata = _extract_qq_metadata(sql_message.content)
                chat_type: Literal["group", "private"] = (
                    "private"
                    if metadata.get("chat_type") == "private"
                    else (
                        "group"
                        if metadata.get("chat_type") == "group"
                        or sql_message.session_id.startswith("group_")
                        else "private"
                    )
                )
                images: list[tuple[ImageReadResult, bool]] = []
                if sql_message.images:
                    # Backward compatibility for old rows. New rows deliberately do
                    # not duplicate image base64; the image description is already
                    # represented in `message`/`content`.
                    images = [
                        (
                            ImageReadResult(
                                base64=item["base64"],
                                phash=imagehash.hex_to_hash(item["phash"]),
                            ),
                            bool(item["flag"]),
                        )
                        for item in json.loads(sql_message.images)
                        if item.get("base64")
                    ]
                user_msg = UserMessage(
                    timestamp=sql_message.timestamp,
                    user=sql_message.user or "",
                    message=QQMessage.from_str(sql_message.message),
                    user_id=sql_message.user_id or "",
                    message_id=sql_message.message_id or "",
                    is_tome=sql_message.is_tome,
                    images=images,
                    chat_type=chat_type,
                    reply_to_id=(
                        str(metadata["reply_to_id"])
                        if metadata.get("reply_to_id")
                        else None
                    ),
                    mention_user_ids=[
                        str(item)
                        for item in metadata.get("mention_user_ids", [])
                        if item
                    ],
                )
                return HumanMessage(
                    content=sql_message.content,
                    additional_kwargs={"raw": user_msg},
                )
            return HumanMessage(content=sql_message.content)
        if role == "ai":
            timestamp = sql_message.created_at
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            return AIMessage(
                content=sql_message.content,
                additional_kwargs={
                    "created_at": timestamp,
                    "history_timestamp": timestamp.isoformat(),
                },
            )
        raise ValueError(f"Unknown message role: {role}")

    @override
    def to_sql_model(self, message: BaseMessage, session_id: str) -> ChatMessageRecord:
        now = datetime.now(UTC)

        if (
            isinstance(message, HumanMessage)
            and (user_msg := message.additional_kwargs.get("raw"))
            and isinstance(user_msg, UserMessage)
        ):
            return ChatMessageRecord(
                session_id=session_id,
                role="human",
                content=content_to_text(message.content),
                created_at=now,
                timestamp=user_msg.timestamp,
                user=user_msg.user,
                user_id=user_msg.user_id,
                message_id=user_msg.message_id,
                is_tome=user_msg.is_tome,
                message=user_msg.message.get_msgcode(),
                images=(
                    json.dumps(
                        [
                            {
                                "phash": str(img.phash),
                                "flag": flag,
                            }
                            for img, flag in user_msg.images
                        ]
                    )
                    if user_msg.images
                    else None
                ),
            )

        if isinstance(message, AIMessage):
            return ChatMessageRecord(
                session_id=session_id,
                role="ai",
                content=content_to_text(message.content),
                created_at=now,
                timestamp=None,
                user=None,
                user_id=None,
                message_id=None,
                is_tome=False,
                message=None,
                images=None,
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
