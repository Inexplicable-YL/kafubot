from __future__ import annotations

# pyright: reportIncompatibleVariableOverride=false
from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar, Literal, get_args, get_origin
from typing_extensions import override

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic.fields import FieldInfo

from kafubot.event import Event

from .message import CQHTTPMessage

DETAIL_TYPE_KEYS = ("message_type", "notice_type", "request_type", "meta_event_type")
EventType = tuple[str | None, str | None, str | None]
EventModels = dict[EventType, type["CQHTTPEvent"]]


def _get_literal_field(field: FieldInfo | None) -> str | None:
    if field is None:
        return None
    annotation = field.annotation
    if annotation is None or get_origin(annotation) is not Literal:
        return None
    literal_values = get_args(annotation)
    if len(literal_values) != 1:
        return None
    return str(literal_values[0])


class CQHTTPEvent(Event):
    """OneBot v11 event base, preserving SekaiBot's public event API."""

    __event__: ClassVar[str] = ""
    type: str | None = Field(alias="post_type")
    time: int
    self_id: int
    post_type: str

    @property
    @override
    def event_name(self) -> str:
        return self.get_event_name()

    def get_type(self) -> str:
        return self.post_type

    @override
    def get_event_name(self) -> str:
        return self.post_type

    @override
    def get_event_description(self) -> str:
        return str(self.model_dump())

    @override
    def get_message(self) -> CQHTTPMessage:
        raise ValueError("event has no message")

    @override
    def get_user_id(self) -> str:
        raise ValueError("event has no user context")

    @override
    def get_session_id(self) -> str:
        raise ValueError("event has no session context")

    @override
    def is_tome(self) -> bool:
        return False

    @classmethod
    def get_event_type(cls) -> EventType:
        post_type = _get_literal_field(cls.model_fields.get("post_type"))
        if post_type is None:
            return (None, None, None)
        return (
            post_type,
            _get_literal_field(cls.model_fields.get(f"{post_type}_type")),
            _get_literal_field(cls.model_fields.get("sub_type")),
        )


class Sender(BaseModel):
    user_id: int | None = None
    nickname: str | None = None
    card: str | None = None
    sex: Literal["male", "female", "unknown"] | None = None
    age: int | None = None
    area: str | None = None
    level: str | None = None
    role: str | None = None
    title: str | None = None


class Reply(BaseModel):
    model_config = ConfigDict(extra="allow")

    time: int
    message_type: str
    message_id: int
    real_id: int
    sender: Sender
    message: CQHTTPMessage


class Anonymous(BaseModel):
    id: int
    name: str
    flag: str


class File(BaseModel):
    id: str
    name: str
    size: int
    busid: int


class Status(BaseModel):
    model_config = ConfigDict(extra="allow")

    online: bool
    good: bool


class MessageEvent(CQHTTPEvent):
    __event__: ClassVar[str] = "message"
    post_type: Literal["message"]
    message_type: Literal["private", "group"]
    sub_type: str
    message_id: int
    user_id: int
    message: CQHTTPMessage
    original_message: CQHTTPMessage = Field(default_factory=CQHTTPMessage)
    raw_message: str = ""
    font: int = 0
    sender: Sender = Field(default_factory=Sender)
    to_me: bool = False
    reply: Reply | None = None

    @model_validator(mode="before")
    @classmethod
    def preserve_original_message(cls, values: Any) -> Any:
        if isinstance(values, dict) and "message" in values:
            values = dict(values)
            values["original_message"] = deepcopy(values["message"])
        return values

    @override
    def get_event_name(self) -> str:
        suffix = f".{self.sub_type}" if self.sub_type else ""
        return f"{self.post_type}.{self.message_type}{suffix}"

    @override
    def get_event_description(self) -> str:
        return f">> MsgID:[{self.message_id}] | User:[{self.user_id}]\n>> " + repr(
            self.original_message
        )

    @override
    def get_message(self) -> CQHTTPMessage:
        return self.message

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)

    @override
    def get_session_id(self) -> str:
        group_id = getattr(self, "group_id", None)
        return (
            f"group_{group_id}_{self.get_user_id()}"
            if group_id
            else self.get_user_id()
        )

    @property
    def conversation_id(self) -> str:
        """Cognitive context partition; unlike legacy session ids, groups are shared."""
        group_id = getattr(self, "group_id", None)
        return f"group_{group_id}" if group_id else f"private_{self.user_id}"

    @override
    def get_plain_text(self) -> str:
        return self.message.get_plain_text()

    @override
    def is_tome(self) -> bool:
        return self.to_me


class PrivateMessageEvent(MessageEvent):
    __event__: ClassVar[str] = "message.private"
    message_type: Literal["private"]
    sub_type: Literal["friend", "group", "other"]


class GroupMessageEvent(MessageEvent):
    __event__: ClassVar[str] = "message.group"
    message_type: Literal["group"]
    sub_type: Literal["normal", "anonymous", "notice"]
    group_id: int
    anonymous: Anonymous | None = None

    @override
    def get_event_description(self) -> str:
        return (
            f">> MsgID:[{self.message_id}] | User:[{self.user_id}] | "
            f"Group:[{self.group_id}]\n>> {self.original_message!r}"
        )


class NoticeEvent(CQHTTPEvent):
    __event__: ClassVar[str] = "notice"
    post_type: Literal["notice"]
    notice_type: str

    @override
    def get_event_name(self) -> str:
        sub_type = getattr(self, "sub_type", None)
        suffix = f".{sub_type}" if sub_type else ""
        return f"{self.post_type}.{self.notice_type}{suffix}"

    @override
    def get_session_id(self) -> str:
        group_id = getattr(self, "group_id", None)
        return (
            f"group_{group_id}_{self.get_user_id()}"
            if group_id
            else self.get_user_id()
        )


class GroupUploadNoticeEvent(NoticeEvent):
    __event__: ClassVar[str] = "notice.group_upload"
    notice_type: Literal["group_upload"]
    user_id: int
    group_id: int
    file: File

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)


class GroupAdminNoticeEvent(NoticeEvent):
    __event__: ClassVar[str] = "notice.group_admin"
    notice_type: Literal["group_admin"]
    sub_type: Literal["set", "unset"]
    user_id: int
    group_id: int

    @override
    def is_tome(self) -> bool:
        return self.user_id == self.self_id

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)


class GroupDecreaseNoticeEvent(NoticeEvent):
    __event__: ClassVar[str] = "notice.group_decrease"
    notice_type: Literal["group_decrease"]
    sub_type: Literal["leave", "kick", "kick_me"]
    group_id: int
    operator_id: int
    user_id: int

    @override
    def is_tome(self) -> bool:
        return self.user_id == self.self_id

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)


class GroupIncreaseNoticeEvent(NoticeEvent):
    __event__: ClassVar[str] = "notice.group_increase"
    notice_type: Literal["group_increase"]
    sub_type: Literal["approve", "invite"]
    group_id: int
    operator_id: int
    user_id: int

    @override
    def is_tome(self) -> bool:
        return self.user_id == self.self_id

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)


class GroupBanNoticeEvent(NoticeEvent):
    __event__: ClassVar[str] = "notice.group_ban"
    notice_type: Literal["group_ban"]
    sub_type: Literal["ban", "lift_ban"]
    group_id: int
    operator_id: int
    user_id: int
    duration: int

    @override
    def is_tome(self) -> bool:
        return self.user_id == self.self_id

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)


class FriendAddNoticeEvent(NoticeEvent):
    __event__: ClassVar[str] = "notice.friend_add"
    notice_type: Literal["friend_add"]
    user_id: int

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)


class GroupRecallNoticeEvent(NoticeEvent):
    __event__: ClassVar[str] = "notice.group_recall"
    notice_type: Literal["group_recall"]
    group_id: int
    operator_id: int
    user_id: int
    message_id: int

    @override
    def is_tome(self) -> bool:
        return self.user_id == self.self_id

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)


class FriendRecallNoticeEvent(NoticeEvent):
    __event__: ClassVar[str] = "notice.friend_recall"
    notice_type: Literal["friend_recall"]
    user_id: int
    message_id: int

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)


class NotifyEvent(NoticeEvent):
    __event__: ClassVar[str] = "notice.notify"
    notice_type: Literal["notify"]
    sub_type: str
    user_id: int
    group_id: int | None

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)


class PokeNotifyEvent(NotifyEvent):
    __event__: ClassVar[str] = "notice.notify.poke"
    sub_type: Literal["poke"]
    target_id: int
    group_id: int | None = None

    @override
    def is_tome(self) -> bool:
        return self.target_id == self.self_id


class GroupLuckyKingNotifyEvent(NotifyEvent):
    __event__: ClassVar[str] = "notice.notify.lucky_king"
    sub_type: Literal["lucky_king"]
    group_id: int
    target_id: int

    @override
    def is_tome(self) -> bool:
        return self.target_id == self.self_id

    @override
    def get_user_id(self) -> str:
        return str(self.target_id)


class GroupHonorNotifyEvent(NotifyEvent):
    __event__: ClassVar[str] = "notice.notify.honor"
    sub_type: Literal["honor"]
    group_id: int
    honor_type: Literal["talkative", "performer", "emotion"]

    @override
    def is_tome(self) -> bool:
        return self.user_id == self.self_id


class RequestEvent(CQHTTPEvent):
    __event__: ClassVar[str] = "request"
    post_type: Literal["request"]
    request_type: str

    @override
    def get_event_name(self) -> str:
        sub_type = getattr(self, "sub_type", None)
        suffix = f".{sub_type}" if sub_type else ""
        return f"{self.post_type}.{self.request_type}{suffix}"

    @override
    def get_session_id(self) -> str:
        group_id = getattr(self, "group_id", None)
        return (
            f"group_{group_id}_{self.get_user_id()}"
            if group_id
            else self.get_user_id()
        )

    async def approve(self) -> dict[str, Any]:
        raise NotImplementedError

    async def refuse(self) -> dict[str, Any]:
        raise NotImplementedError


class FriendRequestEvent(RequestEvent):
    __event__: ClassVar[str] = "request.friend"
    request_type: Literal["friend"]
    user_id: int
    comment: str = ""
    flag: str

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)

    @override
    async def approve(self, remark: str = "") -> dict[str, Any]:
        return await self.adapter.call_api(
            "set_friend_add_request",
            flag=self.flag,
            approve=True,
            remark=remark,
        )

    @override
    async def refuse(self) -> dict[str, Any]:
        return await self.adapter.call_api(
            "set_friend_add_request",
            flag=self.flag,
            approve=False,
        )


class GroupRequestEvent(RequestEvent):
    __event__: ClassVar[str] = "request.group"
    request_type: Literal["group"]
    sub_type: Literal["add", "invite"]
    group_id: int
    user_id: int
    comment: str = ""
    flag: str

    @override
    def get_user_id(self) -> str:
        return str(self.user_id)

    @override
    async def approve(self) -> dict[str, Any]:
        return await self.adapter.call_api(
            "set_group_add_request",
            flag=self.flag,
            sub_type=self.sub_type,
            approve=True,
        )

    @override
    async def refuse(self, reason: str = "") -> dict[str, Any]:
        return await self.adapter.call_api(
            "set_group_add_request",
            flag=self.flag,
            sub_type=self.sub_type,
            approve=False,
            reason=reason,
        )


class MetaEvent(CQHTTPEvent):
    __event__: ClassVar[str] = "meta_event"
    post_type: Literal["meta_event"]
    meta_event_type: str

    @override
    def get_event_name(self) -> str:
        return f"{self.post_type}.{self.meta_event_type}"


class LifecycleMetaEvent(MetaEvent):
    __event__: ClassVar[str] = "meta_event.lifecycle"
    meta_event_type: Literal["lifecycle"]
    sub_type: Literal["enable", "disable", "connect"]


class HeartbeatMetaEvent(MetaEvent):
    __event__: ClassVar[str] = "meta_event.heartbeat"
    meta_event_type: Literal["heartbeat"]
    status: Status
    interval: int


def _default_event_models() -> EventModels:
    models: EventModels = {}
    for value in globals().values():
        if isinstance(value, type) and issubclass(value, CQHTTPEvent):
            models[value.get_event_type()] = value
    return models


DEFAULT_EVENT_MODELS = _default_event_models()


def get_event_model(
    payload: Mapping[str, Any],
    event_models: Mapping[EventType, type[CQHTTPEvent]] | None = None,
) -> type[CQHTTPEvent]:
    models = event_models or DEFAULT_EVENT_MODELS
    post_type = payload.get("post_type")
    if not isinstance(post_type, str):
        return models.get((None, None, None), CQHTTPEvent)
    detail_type: str | None = None
    for key in DETAIL_TYPE_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            detail_type = value
            break
    if detail_type is None:
        value = payload.get(f"{post_type}_type")
        detail_type = value if isinstance(value, str) else None
    sub_type = payload.get("sub_type")
    normalized_sub_type = sub_type if isinstance(sub_type, str) else None
    return (
        models.get((post_type, detail_type, normalized_sub_type))
        or models.get((post_type, detail_type, None))
        or models.get((post_type, None, None))
        or models.get((None, None, None), CQHTTPEvent)
    )


def parse_event(
    adapter: Any,
    payload: dict[str, Any],
    event_models: Mapping[EventType, type[CQHTTPEvent]] | None = None,
) -> CQHTTPEvent:
    model = get_event_model(payload, event_models)
    return model.model_validate({**payload, "adapter": adapter})


__all__ = [
    "Anonymous",
    "CQHTTPEvent",
    "DEFAULT_EVENT_MODELS",
    "DETAIL_TYPE_KEYS",
    "EventModels",
    "File",
    "FriendAddNoticeEvent",
    "FriendRecallNoticeEvent",
    "FriendRequestEvent",
    "GroupAdminNoticeEvent",
    "GroupBanNoticeEvent",
    "GroupDecreaseNoticeEvent",
    "GroupHonorNotifyEvent",
    "GroupIncreaseNoticeEvent",
    "GroupLuckyKingNotifyEvent",
    "GroupMessageEvent",
    "GroupRecallNoticeEvent",
    "GroupRequestEvent",
    "GroupUploadNoticeEvent",
    "HeartbeatMetaEvent",
    "LifecycleMetaEvent",
    "MessageEvent",
    "MetaEvent",
    "NoticeEvent",
    "NotifyEvent",
    "PokeNotifyEvent",
    "PrivateMessageEvent",
    "Reply",
    "RequestEvent",
    "Sender",
    "Status",
    "get_event_model",
    "parse_event",
]
