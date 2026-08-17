from __future__ import annotations

# pyright: reportIncompatibleVariableOverride=false
import json
from typing import TYPE_CHECKING, Any, Literal, cast

import anyio
import pytest
from anyio.lowlevel import checkpoint

from kafubot.adapters.cqhttp import CQHTTPAdapter
from kafubot.adapters.cqhttp.event import (
    CQHTTPEvent,
    FriendAddNoticeEvent,
    FriendRecallNoticeEvent,
    FriendRequestEvent,
    GroupAdminNoticeEvent,
    GroupBanNoticeEvent,
    GroupDecreaseNoticeEvent,
    GroupHonorNotifyEvent,
    GroupIncreaseNoticeEvent,
    GroupLuckyKingNotifyEvent,
    GroupMessageEvent,
    GroupRecallNoticeEvent,
    GroupRequestEvent,
    GroupUploadNoticeEvent,
    HeartbeatMetaEvent,
    LifecycleMetaEvent,
    NoticeEvent,
    PokeNotifyEvent,
    parse_event,
)
from kafubot.adapters.cqhttp.exceptions import (
    ActionFailed,
    CQHTTPError,
    CQHTTPException,
)
from kafubot.adapters.cqhttp.message import CQHTTPMessage, CQHTTPMessageSegment
from kafubot.config import AppConfig, CQHTTPConfig

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from kafubot.bot import Bot
    from kafubot.protocol import Event


def group_payload(message: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "time": 1_700_000_000,
        "self_id": 10001,
        "post_type": "message",
        "message_type": "group",
        "sub_type": "normal",
        "message_id": 42,
        "group_id": 20002,
        "user_id": 30003,
        "message": message,
        "raw_message": "",
        "font": 0,
        "sender": {"user_id": 30003, "nickname": "测试用户"},
    }


class FakeBot:
    def __init__(
        self,
        config: CQHTTPConfig,
        receive: Callable[[Event], Awaitable[None]],
    ) -> None:
        self.config = AppConfig(adapter={"cqhttp": config.model_dump()})
        self._should_exit = anyio.Event()
        self.receive = receive

    async def submit_event(self, event: Event) -> None:
        await self.receive(event)


def make_adapter(
    config: CQHTTPConfig,
    receive: Callable[[Event], Awaitable[None]],
) -> CQHTTPAdapter:
    return CQHTTPAdapter(cast("Bot", FakeBot(config, receive)))


def test_message_container_supports_onebot_segments() -> None:
    message = CQHTTPMessageSegment.reply(12) + CQHTTPMessageSegment.at(34) + " hello"

    assert isinstance(message, CQHTTPMessage)
    assert [segment.type for segment in message] == ["reply", "at", "text"]
    assert message.get_plain_text() == " hello"
    assert str(message[0]) == "[CQ:reply,id=12]"


def test_parse_event_preserves_original_message() -> None:
    payload = group_payload([{"type": "text", "data": {"text": "hello"}}])
    event = parse_event(object(), payload)

    assert isinstance(event, GroupMessageEvent)
    event.message[0].data["text"] = "changed"
    assert event.original_message[0].data["text"] == "hello"
    assert event.get_session_id() == "group_20002_30003"
    assert event.conversation_id == "group_20002"


@pytest.mark.parametrize(
    ("event_class", "payload"),
    [
        (
            GroupUploadNoticeEvent,
            {
                "notice_type": "group_upload",
                "group_id": 2,
                "user_id": 3,
                "file": {"id": "f", "name": "a.txt", "size": 4, "busid": 5},
            },
        ),
        (
            GroupAdminNoticeEvent,
            {
                "notice_type": "group_admin",
                "sub_type": "set",
                "group_id": 2,
                "user_id": 3,
            },
        ),
        (
            GroupDecreaseNoticeEvent,
            {
                "notice_type": "group_decrease",
                "sub_type": "leave",
                "group_id": 2,
                "operator_id": 3,
                "user_id": 4,
            },
        ),
        (
            GroupIncreaseNoticeEvent,
            {
                "notice_type": "group_increase",
                "sub_type": "approve",
                "group_id": 2,
                "operator_id": 3,
                "user_id": 4,
            },
        ),
        (
            GroupBanNoticeEvent,
            {
                "notice_type": "group_ban",
                "sub_type": "ban",
                "group_id": 2,
                "operator_id": 3,
                "user_id": 4,
                "duration": 60,
            },
        ),
        (FriendAddNoticeEvent, {"notice_type": "friend_add", "user_id": 3}),
        (
            GroupRecallNoticeEvent,
            {
                "notice_type": "group_recall",
                "group_id": 2,
                "operator_id": 3,
                "user_id": 4,
                "message_id": 5,
            },
        ),
        (
            FriendRecallNoticeEvent,
            {"notice_type": "friend_recall", "user_id": 3, "message_id": 5},
        ),
        (
            PokeNotifyEvent,
            {
                "notice_type": "notify",
                "sub_type": "poke",
                "user_id": 3,
                "target_id": 10001,
            },
        ),
        (
            GroupLuckyKingNotifyEvent,
            {
                "notice_type": "notify",
                "sub_type": "lucky_king",
                "group_id": 2,
                "user_id": 3,
                "target_id": 4,
            },
        ),
        (
            GroupHonorNotifyEvent,
            {
                "notice_type": "notify",
                "sub_type": "honor",
                "group_id": 2,
                "user_id": 3,
                "honor_type": "talkative",
            },
        ),
        (
            FriendRequestEvent,
            {
                "post_type": "request",
                "request_type": "friend",
                "user_id": 3,
                "comment": "hello",
                "flag": "friend-flag",
            },
        ),
        (
            GroupRequestEvent,
            {
                "post_type": "request",
                "request_type": "group",
                "sub_type": "invite",
                "group_id": 2,
                "user_id": 3,
                "comment": "hello",
                "flag": "group-flag",
            },
        ),
        (
            LifecycleMetaEvent,
            {
                "post_type": "meta_event",
                "meta_event_type": "lifecycle",
                "sub_type": "connect",
            },
        ),
        (
            HeartbeatMetaEvent,
            {
                "post_type": "meta_event",
                "meta_event_type": "heartbeat",
                "status": {"online": True, "good": True},
                "interval": 5000,
            },
        ),
    ],
)
def test_complete_onebot_event_models(
    event_class: type[CQHTTPEvent],
    payload: dict[str, Any],
) -> None:
    complete_payload = {
        "time": 1_700_000_000,
        "self_id": 10001,
        "post_type": "notice",
        **payload,
    }

    assert type(parse_event(object(), complete_payload)) is event_class


class CustomNoticeEvent(NoticeEvent):
    notice_type: Literal["custom_notice"]
    value: str


def test_adapter_supports_custom_event_model_registration() -> None:
    key = CustomNoticeEvent.get_event_type()
    previous = CQHTTPAdapter.event_models.get(key)
    try:
        CQHTTPAdapter.add_event_model(CustomNoticeEvent)
        model = CQHTTPAdapter.get_event_model("notice", "custom_notice", None)
        event = parse_event(
            object(),
            {
                "time": 1,
                "self_id": 2,
                "post_type": "notice",
                "notice_type": "custom_notice",
                "value": "ok",
            },
            CQHTTPAdapter.event_models,
        )
        assert model is CustomNoticeEvent
        assert isinstance(event, CustomNoticeEvent)
        assert event.value == "ok"
    finally:
        if previous is None:
            CQHTTPAdapter.event_models.pop(key, None)
        else:
            CQHTTPAdapter.event_models[key] = previous


def test_complete_onebot_message_segment_builders() -> None:
    segments = [
        CQHTTPMessageSegment.rps(),
        CQHTTPMessageSegment.dice(),
        CQHTTPMessageSegment.shake(),
        CQHTTPMessageSegment.poke("poke", 1),
        CQHTTPMessageSegment.anonymous(ignore=True),
        CQHTTPMessageSegment.share("https://example.com", "title"),
        CQHTTPMessageSegment.contact_friend(2),
        CQHTTPMessageSegment.contact_group(3),
        CQHTTPMessageSegment.location(1.5, 2.5, "place"),
        CQHTTPMessageSegment.music("qq", 4),
        CQHTTPMessageSegment.music_custom("url", "audio", "title"),
    ]

    assert [segment.type for segment in segments] == [
        "rps",
        "dice",
        "shake",
        "poke",
        "anonymous",
        "share",
        "contact",
        "contact",
        "location",
        "music",
        "music",
    ]
    assert segments[6].data == {"type": "qq", "id": "2"}
    assert segments[10].data["type"] == "custom"
    assert str(CQHTTPMessageSegment.text("a&[b]")) == "a&[b]"
    assert CQHTTPMessageSegment.text("a&[b]").get_cqcode() == "a&amp;&#91;b&#93;"


def test_sekaibot_exception_compatibility() -> None:
    response = {"status": "failed", "retcode": 1}
    error = ActionFailed(resp=response)

    assert CQHTTPError is CQHTTPException
    assert error.resp is response
    assert error.response is response


@pytest.mark.anyio
async def test_adapter_extracts_at_me_before_dispatch() -> None:
    received: list[GroupMessageEvent] = []

    async def receive(event: Any) -> None:
        received.append(event)

    adapter = make_adapter(CQHTTPConfig(), receive)
    payload = group_payload(
        [
            {"type": "at", "data": {"qq": "10001"}},
            {"type": "text", "data": {"text": "  你好"}},
        ]
    )

    await adapter.handle_cqhttp_event(payload)

    assert len(received) == 1
    assert received[0].is_tome()
    assert received[0].message.get_plain_text() == "你好"


class FakeWebSocket:
    closed = False

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_str(self, data: str) -> None:
        self.sent.append(json.loads(data))


@pytest.mark.anyio
async def test_concurrent_api_calls_are_matched_by_echo() -> None:
    async def receive(_event: Any) -> None:
        return None

    adapter = make_adapter(CQHTTPConfig(api_timeout=1), receive)
    websocket = FakeWebSocket()
    adapter.websocket = websocket  # type: ignore[assignment]
    results: dict[str, Any] = {}

    async def invoke(name: str) -> None:
        results[name] = await adapter.call_api(name, value=name)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(invoke, "first")
        task_group.start_soon(invoke, "second")
        while len(websocket.sent) < 2:
            await checkpoint()
        for request in reversed(websocket.sent):
            await adapter._handle_text(
                json.dumps(
                    {
                        "status": "ok",
                        "retcode": 0,
                        "data": request["params"]["value"],
                        "echo": request["echo"],
                    }
                )
            )

    assert results == {"first": "first", "second": "second"}
