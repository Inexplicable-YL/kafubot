import itertools
import os
from datetime import datetime  # noqa: TC003
from html import escape
from operator import add
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired
from typing_extensions import TypedDict
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain.agents import AgentState
from pydantic import BaseModel, ConfigDict, Field

from chat.image import ImageReadResult  # noqa: TC001
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

    def as_plain_content(self) -> str:
        msg = QQMessage(filter(lambda x: x.type != "reply", self.message))
        return f"{escape(self.user, quote=True)}: {msg.get_msgcode()}"


class OutputMessage(TypedDict):
    type: Literal["reply", "finish", "stop", "meme"]
    data: dict[str, Any]


class ManagerState(AgentState, extra_items=Any):
    inputs: list[UserMessage]
    outputs: Annotated[list[OutputMessage], add]

    early_messages: list[BaseMessage]
    full_messages: list[BaseMessage]

    meme_id: NotRequired[itertools.count]

    group_id: str
    user_map: dict[str, str]

    real_average_count: NotRequired[float]
    real_meme_ratio: NotRequired[float]


class ManagerContext(TypedDict):
    session_id: str
    talk_value: float
    average_reply_count: float
    meme_reply_ratio: float
    is_tome: bool
    node: Any
