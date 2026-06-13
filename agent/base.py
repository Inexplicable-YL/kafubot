import os
from datetime import datetime  # noqa: TC003
from html import escape
from operator import add
from typing import Annotated, Any, Literal, NotRequired, TypedDict
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langchain.agents import AgentState
from langchain_core.messages import AnyMessage, BaseMessage
from pydantic import BaseModel, ConfigDict, Field

from agent.message import QQMessage  # noqa: TC001
from agent.multimodal.image import ImageReadResult  # noqa: TC001

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
    images: list[tuple[ImageReadResult, bool]] = Field(default_factory=list)

    def as_content(self) -> str:
        return f"<user-message message_id={escape(self.message_id, quote=True)}, user={escape(self.user, quote=True)}>\n{self.message.get_msgcode()}\n</user-message>"

    def as_plain_content(self) -> str:
        msg = QQMessage(filter(lambda x: x.type != "reply", self.message))
        return f"{escape(self.user, quote=True)}: {msg.get_msgcode()}"


class OutputMessage(TypedDict):
    type: Literal["reply", "finish", "stop", "meme", "limit"]
    data: dict[str, Any]


class ManagerState(AgentState):
    inputs: list[UserMessage]
    outputs: Annotated[list[OutputMessage], add]
    currents: list[AnyMessage]
    histories: list[AnyMessage]

    summary_pruned_messages: NotRequired[list[BaseMessage] | None]

    reply_top_messages: NotRequired[Annotated[list[BaseMessage], add]]
    reply_bottom_messages: NotRequired[Annotated[list[BaseMessage], add]]


class ManagerContext(TypedDict):
    node: Any
    session_id: str
    is_tome: bool
    unrestricted: bool
