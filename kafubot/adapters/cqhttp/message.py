from __future__ import annotations

from typing import Literal, Self
from typing_extensions import override

from kafubot.message import Message, MessageSegment

__all__ = ["CQHTTPMessage", "CQHTTPMessageSegment", "escape"]


class CQHTTPMessage(Message["CQHTTPMessageSegment"]):
    @classmethod
    @override
    def get_segment_class(cls) -> type[CQHTTPMessageSegment]:
        return CQHTTPMessageSegment


class CQHTTPMessageSegment(MessageSegment[CQHTTPMessage]):
    @classmethod
    @override
    def get_message_class(cls) -> type[CQHTTPMessage]:
        return CQHTTPMessage

    @classmethod
    @override
    def from_str(cls, msg: str) -> Self:
        return cls.text(msg)

    @override
    def __str__(self) -> str:
        if self.type == "text":
            return str(self.data.get("text", ""))
        return self.get_cqcode()

    def get_cqcode(self) -> str:
        if self.type == "text":
            return escape(str(self.data.get("text", "")), escape_comma=False)
        params = ",".join(
            f"{key}={escape(str(value))}"
            for key, value in self.data.items()
            if value is not None
        )
        return f"[CQ:{self.type}{',' if params else ''}{params}]"

    @classmethod
    def text(cls, text: str) -> Self:
        return cls(type="text", data={"text": text})

    @classmethod
    def face(cls, id_: int) -> Self:
        return cls(type="face", data={"id": str(id_)})

    @classmethod
    def image(
        cls,
        file: str,
        type_: Literal["flash"] | None = None,
        sub_type: Literal[0, 1] = 0,
        cache: bool = True,
        proxy: bool = True,
        timeout: int | None = None,
    ) -> Self:
        return cls(
            type="image",
            data={
                "file": file,
                "type": type_,
                "sub_type": sub_type,
                "cache": cache,
                "proxy": proxy,
                "timeout": timeout,
            },
        )

    @classmethod
    def record(
        cls,
        file: str,
        magic: bool = False,
        cache: bool = True,
        proxy: bool = True,
        timeout: int | None = None,
    ) -> Self:
        return cls(
            type="record",
            data={
                "file": file,
                "magic": magic,
                "cache": cache,
                "proxy": proxy,
                "timeout": timeout,
            },
        )

    @classmethod
    def video(
        cls,
        file: str,
        cache: bool = True,
        proxy: bool = True,
        timeout: int | None = None,
    ) -> Self:
        return cls(
            type="video",
            data={"file": file, "cache": cache, "proxy": proxy, "timeout": timeout},
        )

    @classmethod
    def at(cls, qq: int | Literal["all"]) -> Self:
        return cls(type="at", data={"qq": str(qq)})

    @classmethod
    def rps(cls) -> Self:
        return cls(type="rps", data={})

    @classmethod
    def dice(cls) -> Self:
        return cls(type="dice", data={})

    @classmethod
    def shake(cls) -> Self:
        return cls(type="shake", data={})

    @classmethod
    def poke(cls, type_: str, id_: int) -> Self:
        return cls(type="poke", data={"type": type_, "id": str(id_)})

    @classmethod
    def anonymous(cls, ignore: bool | None = None) -> Self:
        return cls(type="anonymous", data={"ignore": ignore})

    @classmethod
    def share(
        cls,
        url: str,
        title: str,
        content: str | None = None,
        image: str | None = None,
    ) -> Self:
        return cls(
            type="share",
            data={"url": url, "title": title, "content": content, "image": image},
        )

    @classmethod
    def contact(cls, type_: Literal["qq", "group"], id_: int) -> Self:
        return cls(type="contact", data={"type": type_, "id": str(id_)})

    @classmethod
    def contact_friend(cls, id_: int) -> Self:
        return cls.contact("qq", id_)

    @classmethod
    def contact_group(cls, id_: int) -> Self:
        return cls.contact("group", id_)

    @classmethod
    def location(
        cls,
        lat: float,
        lon: float,
        title: str | None,
        content: str | None = None,
    ) -> Self:
        return cls(
            type="location",
            data={
                "lat": str(lat),
                "lon": str(lon),
                "title": title,
                "content": content,
            },
        )

    @classmethod
    def music(cls, type_: Literal["qq", "163", "xm"], id_: int) -> Self:
        return cls(type="music", data={"type": type_, "id": str(id_)})

    @classmethod
    def music_custom(
        cls,
        url: str,
        audio: str,
        title: str,
        content: str | None = None,
        image: str | None = None,
    ) -> Self:
        return cls(
            type="music",
            data={
                "type": "custom",
                "url": url,
                "audio": audio,
                "title": title,
                "content": content,
                "image": image,
            },
        )

    @classmethod
    def reply(cls, id_: int) -> Self:
        return cls(type="reply", data={"id": str(id_)})

    @classmethod
    def node(cls, id_: int) -> Self:
        return cls(type="node", data={"id": str(id_)})

    @classmethod
    def node_custom(
        cls,
        user_id: int,
        nickname: str,
        content: CQHTTPMessage,
    ) -> Self:
        return cls(
            type="node",
            data={
                "user_id": str(user_id),
                "nickname": nickname,
                "content": content,
            },
        )

    @classmethod
    def json_message(cls, data: str) -> Self:
        return cls(type="json", data={"data": data})

    @classmethod
    def xml_message(cls, data: str) -> Self:
        return cls(type="xml", data={"data": data})


def escape(string: str, *, escape_comma: bool = True) -> str:
    string = string.replace("&", "&amp;").replace("[", "&#91;").replace("]", "&#93;")
    if escape_comma:
        string = string.replace(",", "&#44;")
    return string
