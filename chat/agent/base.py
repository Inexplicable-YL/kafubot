from __future__ import annotations

import os
from datetime import datetime, tzinfo  # noqa: TC003
from html import escape
from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import TypedDict
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain.agents import AgentState
from pydantic import BaseModel, ConfigDict, Field

from chat.image import ImageReadResult  # noqa: TC001
from chat.meme import MemeResult  # noqa: TC001
from chat.message import QQMessage  # noqa: TC001

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
else:
    BaseMessage = Any

load_dotenv()


MODEL_VISIBLE_TZ = ZoneInfo(os.getenv("MODEL_VISIBLE_TZ", "Asia/Shanghai"))


class UserMessage(BaseModel):
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

    def as_content(self) -> str:
        return f"<user-message message_id={escape(self.message_id, quote=True)}, user={escape(self.user, quote=True)}>\n{self.message.get_msgcode()}\n</user-message>"

    def as_plain_content(self, *, timezone: tzinfo = MODEL_VISIBLE_TZ) -> str:
        return f"[{self.timestamp.astimezone(timezone).strftime('%Y-%m-%d %H:%M:%S')}]{escape(self.user, quote=True)}: {self.message.get_msgcode()}"


class OutputMessage(TypedDict):
    type: Literal["reply", "finish", "stop", "meme"]
    data: dict[str, Any]


class ManagerState(AgentState):
    inputs: list[UserMessage]
    early_messages: list[BaseMessage]
    full_messages: list[BaseMessage]
    reasoning_effort: Literal["high", "max"]
    should_stop: bool
    outputs: list[OutputMessage]
    search_meme_history: dict[int, MemeResult]
    meme_id: int
    deferred_tools: list[Any]


class ManagerContext(TypedDict):
    session_id: str
    is_tome: bool
    node: Any
