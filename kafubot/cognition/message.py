import json
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import Any, Protocol, Self, cast
from typing_extensions import override
from zoneinfo import ZoneInfo

from kafubot.adapters.cqhttp.message import CQHTTPMessage, CQHTTPMessageSegment
from kafubot.cognition.media.image import ImageReadResult
from kafubot.cognition.plugins.meme.backend import search_meme
from kafubot.message import Message, MessageSegment

__all__ = ["QQMessage", "QQMessageSegment", "escape"]


class HistoryMessage(Protocol):
    user: str
    user_id: str
    timestamp: datetime
    message_id: str


class QQMessage(Message["QQMessageSegment"]):
    """MSG 消息。"""

    @override
    @classmethod
    def get_segment_class(cls) -> type["QQMessageSegment"]:
        return QQMessageSegment

    @classmethod
    def from_str(cls, msg: str) -> "QQMessage":
        return parse_message(msg)

    def get_msgcode(self) -> str:
        return "".join(seg.get_msgcode() for seg in self)

    async def get_cqhttp_message(
        self, messages: Sequence[HistoryMessage]
    ) -> CQHTTPMessage:
        cq_message = CQHTTPMessage()
        for segment in self:
            if segment := await segment.get_cqhttp_segment(messages):
                if segment.type == "reply" and any(
                    seg.type == "reply" for seg in cq_message
                ):
                    continue
                cq_message += segment
                if segment.type == "at":
                    cq_message += " "
        return cq_message

    @classmethod
    async def from_cqhttp_message(
        cls,
        message: CQHTTPMessage,
        messages: Sequence[HistoryMessage],
        get_image: Callable[[str], Awaitable[ImageReadResult | None]],
    ) -> "QQMessage":
        return cls(
            filter(
                None,
                [
                    await QQMessageSegment.from_cqhttp_segment(
                        segment, messages, get_image
                    )
                    for segment in message
                ],
            )
        )


class QQMessageSegment(MessageSegment["QQMessage"]):
    """MSG 消息字段。"""

    include: set[str] = set()

    @override
    @classmethod
    def get_message_class(cls) -> type[QQMessage]:
        return QQMessage

    @override
    @classmethod
    def from_str(cls, msg: str) -> Self:
        return cls.text(msg)

    @override
    def __str__(self) -> str:
        if self.type == "text":
            return self.data.get("text", "")
        return self.get_msgcode()

    def get_msgcode(self) -> str:
        """获取此消息字段的 MSG 码形式。

        Returns:
            此消息字段的 MSG 码形式。
        """
        if self.type == "text":
            return escape(self.data.get("text", ""), escape_comma=False)

        data = self.data
        if self.include:
            data = {
                key: value for key, value in self.data.items() if key in self.include
            }
        params = ",".join(
            [f"{k}={escape(str(v))}" for k, v in data.items() if v is not None]
        )
        return f"[MSG:{self.type}{',' if params else ''}{params}]"

    @classmethod
    async def from_cqhttp_segment(
        cls,
        segment: CQHTTPMessageSegment,
        messages: Sequence[HistoryMessage],
        get_image: Callable[[str], Awaitable[ImageReadResult | None]],
    ) -> "QQMessageSegment | None":
        if segment.type == "text":
            return QQMessageSegment.text(text=segment.data.get("text", ""))
        if segment.type == "image" and (file := segment.data.get("file")):
            as_meme = str(segment.data.get("sub_type", "0")) == "1"
            if image := await get_image(file):
                if as_meme:
                    return QQMessageSegment.meme(image=image)
                return QQMessageSegment.image(image=image)
        if segment.type == "file" and (file_name := segment.data.get("file")):
            return QQMessageSegment.file(file=file_name)
        if segment.type == "json" and (
            data := json.loads(segment.data.get("data", "{}"))
        ):
            return QQMessageSegment.link(prompt=data.get("prompt", ""), data=data)
        if segment.type == "at" and (qq_number := str(segment.data.get("qq", ""))):
            at_name: str | None = None
            for item in messages:
                if str(item.user_id) == qq_number:
                    at_name = item.user
                    break
            return QQMessageSegment.at(name=at_name or "", user_id=qq_number)
        if segment.type == "reply" and (
            reply_id := str(segment.data.get("id", "")).strip()
        ):
            return QQMessageSegment.reply(
                message_id=reply_id,
                include={"message_id"},
            )
        if segment.type == "forward" and (id_ := str(segment.data.get("id", ""))):
            return QQMessageSegment(type="forward", data={"id": id_})
        return None

    async def get_cqhttp_segment(
        self, messages: Sequence[HistoryMessage]
    ) -> CQHTTPMessageSegment | None:
        """将此消息字段转换为 CQHTTP 消息段。

        Returns:
            转换后的 CQHTTP 消息段。
        """
        if self.type == "text":
            return CQHTTPMessageSegment.text(self.data.get("text", ""))
        if self.type == "meme":
            file = self.data.get("file")
            if (not file) and (image := self.data.get("image")):
                file = cast("ImageReadResult", image).base64
            if (
                (not file)
                and (content := self.data.get("content", ""))
                and (
                    meme_result := await search_meme(
                        content, temperature=0.5, min_score=0.0
                    )
                )
            ):
                file = meme_result.base64
            if file:
                return CQHTTPMessageSegment.image(file, sub_type=1)
        elif self.type == "at":
            if user_id := str(self.data.get("user_id", "")).strip():
                if not any(str(item.user_id) == user_id for item in messages):
                    return None
                try:
                    return CQHTTPMessageSegment.at(int(user_id))
                except ValueError:
                    return None
            name = self.data.get("name", "")
            for item in messages:
                if item.user == name:
                    user_id = int(item.user_id)
                    return CQHTTPMessageSegment.at(user_id)
        elif self.type == "reply":
            if message_id := str(self.data.get("message_id", "")).strip():
                if not any(str(item.message_id) == message_id for item in messages):
                    return None
                try:
                    return CQHTTPMessageSegment.reply(int(message_id))
                except ValueError:
                    return None
            time = self.data.get("time", "")
            for item in messages:
                t = cast("datetime", item.timestamp).astimezone(
                    ZoneInfo("Asia/Shanghai")
                )
                if t.strftime("%H:%M:%S") in time and (message_id := item.message_id):
                    return CQHTTPMessageSegment.reply(int(message_id))
        return None

    @classmethod
    def text(cls, text: str) -> Self:
        """纯文本"""
        return cls(type="text", data={"text": text})

    @classmethod
    def meme(
        cls,
        content: str | None = None,
        file: str | None = None,
        image: ImageReadResult | None = None,
    ) -> Self:
        """表情包"""
        if not (content or file or image):
            raise ValueError("content or file or image must be provided")
        return cls(
            type="meme",
            data={"content": content, "file": file, "image": image},
            include={"content"},
        )

    @classmethod
    def image(
        cls,
        content: str | None = None,
        file: str | None = None,
        image: ImageReadResult | None = None,
    ) -> Self:
        """图片"""
        if not (content or file or image):
            raise ValueError("content or file or image must be provided")
        return cls(
            type="image",
            data={"content": content, "file": file, "image": image},
            include={"content"},
        )

    @classmethod
    def at(cls, name: str = "", user_id: str | None = None) -> Self:
        """@某人"""
        return cls(
            type="at",
            data={"name": name, "user_id": user_id},
            include={"name", "user_id"},
        )

    @classmethod
    def reply(
        cls,
        time: str | None = None,
        message_id: str | None = None,
        *,
        include: set[str] | None = None,
    ) -> Self:
        """回复"""
        return cls(
            type="reply",
            data={"time": time, "message_id": message_id},
            include=include or {"time"},
        )

    @classmethod
    def file(cls, file: str) -> Self:
        return cls(type="file", data={"file": file})

    @classmethod
    def link(cls, prompt: str, data: str | dict[Any, Any]) -> Self:
        if isinstance(data, dict):
            data = json.dumps(data)
        return cls(
            type="json",
            data={"prompt": prompt, "data": data},
            include={"prompt"},
        )


def escape(string: str, *, escape_comma: bool = True) -> str:
    """对 MSG 码中的特殊字符进行转义。

    Args:
        string: 待转义的字符串。
        escape_comma: 是否转义 `,`。

    Returns:
        转义后的字符串。
    """
    string = string.replace("&", "&amp;").replace("[", "&#91;").replace("]", "&#93;")
    if escape_comma:
        string = string.replace(",", "&#44;")
    return string


def parse_message(text: str) -> QQMessage:
    text = text.replace("：", ":").strip()
    segments = QQMessage()
    pattern = re.compile(r"\[MSG:[^\]]*\]")

    def replacer(match: re.Match) -> str:
        nonlocal segments
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
        seg = QQMessageSegment(type=typ, data=data)
        if seg:
            segments += seg
            return ""
        return block

    return segments + pattern.sub(replacer, text).strip()
