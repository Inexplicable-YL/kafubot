import json
import os
from datetime import datetime  # noqa: TC003
from html import escape
from typing import Any, Literal
from typing_extensions import override
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from kafubot.cognition.media.image import ImageReadResult  # noqa: TC001
from kafubot.cognition.message import QQMessage  # noqa: TC001

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
    chat_type: Literal["group", "private"] = "group"
    reply_to_id: str | None = None
    mention_user_ids: list[str] = Field(default_factory=list)

    @override
    def model_post_init(self, __context: Any, /) -> None:
        """Derive only transport facts that are present in QQ message segments."""
        if self.reply_to_id is None:
            self.reply_to_id = next(
                (
                    str(segment.data.get("message_id") or segment.data.get("id"))
                    for segment in self.message
                    if segment.type == "reply"
                    and (segment.data.get("message_id") or segment.data.get("id"))
                ),
                None,
            )
        if not self.mention_user_ids:
            self.mention_user_ids = list(
                dict.fromkeys(
                    str(segment.data["user_id"])
                    for segment in self.message
                    if segment.type == "at" and segment.data.get("user_id")
                )
            )

    def as_content(self) -> str:
        metadata = json.dumps(
            {
                "message_id": self.message_id,
                "user_id": self.user_id,
                "user": self.user,
                "chat_type": self.chat_type,
                "directed_to_bot": self.is_tome,
                "reply_to_id": self.reply_to_id,
                "mention_user_ids": self.mention_user_ids,
                "timestamp": self.timestamp.isoformat(),
            },
            ensure_ascii=False,
        )
        return (
            f"<qq-user-message metadata={escape(metadata, quote=True)!r}>\n"
            f"{self.message.get_msgcode()}\n</qq-user-message>"
        )

    def as_plain_content(self) -> str:
        msg = QQMessage(filter(lambda x: x.type != "reply", self.message))
        return f"{escape(self.user, quote=True)}: {msg.get_msgcode()}"
