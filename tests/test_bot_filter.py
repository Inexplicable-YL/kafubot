from __future__ import annotations

from typing import Any

import anyio
import pytest

from kafubot.adapters.cqhttp.event import (
    FriendAddNoticeEvent,
    GroupMessageEvent,
    PrivateMessageEvent,
    parse_event,
)
from kafubot.bot import Bot


def payload(post_type: str, **fields: Any) -> dict[str, Any]:
    return {
        "time": 1_700_000_000,
        "self_id": 10001,
        "post_type": post_type,
        **fields,
    }


@pytest.mark.anyio
async def test_bot_queue_is_a_single_chat_event_filter() -> None:
    bot = Bot.__new__(Bot)
    bot._event_send, receive = anyio.create_memory_object_stream[
        GroupMessageEvent | PrivateMessageEvent
    ](1)
    notice = parse_event(
        object(),
        payload("notice", notice_type="friend_add", user_id=2),
    )
    message = parse_event(
        object(),
        payload(
            "message",
            message_type="group",
            sub_type="normal",
            message_id=3,
            group_id=4,
            user_id=2,
            message=[{"type": "text", "data": {"text": "hello"}}],
            raw_message="hello",
            font=0,
            sender={"user_id": 2},
        ),
    )
    assert isinstance(notice, FriendAddNoticeEvent)
    assert isinstance(message, GroupMessageEvent)

    await bot.submit_event(notice)
    with pytest.raises(anyio.WouldBlock):
        receive.receive_nowait()

    await bot.submit_event(message)
    assert await receive.receive() is message

    await bot._event_send.aclose()
    await receive.aclose()
