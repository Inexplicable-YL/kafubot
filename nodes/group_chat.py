from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from typing_extensions import override
from zoneinfo import ZoneInfo

import anyio
from _group_chat import (
    AssistantReply,
    get_agent_app,
)
from _prompt import (
    MORE_SENTENCE_REPLY_PROMPT,
    ONE_SENTENCE_REPLY_PROMPT,
    get_extra_prompt,
)
from langchain_core.runnables import Runnable
from sekaibot import Node
from sekaibot.adapter.cqhttp.event import GroupMessageEvent

INTERVAL_SECONDS = 40
ACTIVITY_LIMITS: tuple[tuple[int, int], ...] = (  # (window_seconds, threshold)
    (300, 15),
    (3600, 30),
    (3600 * 5, 100),
    (3600 * 24 * 7, 400),
)
ACTIVITY_HISTORY_LIMIT = max((threshold for _, threshold in ACTIVITY_LIMITS), default=0)

ALLOWED_AUTO_REPLY_GROUPS = {
    596488203,
    1011357049,
    1058218429,
    1087911123,
    834922207,
    788499440,
}


def _is_activity_limited(activites: list[datetime], event_time: int) -> bool:
    return any(
        threshold > 0
        and window_seconds > 0
        and len(activites) >= threshold
        and event_time - int(activites[-threshold].timestamp()) < window_seconds
        for window_seconds, threshold in ACTIVITY_LIMITS
    )


@dataclass
class Histories:
    messages: list[dict[str, Any]] = field(default_factory=list)
    messages_for_extra_prompt: list[dict[str, Any]] = field(default_factory=list)
    activites: list[datetime] = field(default_factory=list)
    timestamp: int = 0
    on_handle: bool = False
    lock: anyio.Lock = field(default_factory=anyio.Lock)


@dataclass
class GroupChatState:
    agent: Runnable[dict[str, Any], None | AssistantReply] = field(
        default_factory=get_agent_app
    )
    storages: dict[str, Histories] = field(default_factory=dict)
    storages_lock: anyio.Lock = field(default_factory=anyio.Lock)


class GroupChat(Node[GroupMessageEvent, GroupChatState, Any]):
    """群聊记录节点"""

    priority = 1

    @override
    def __init_state__(self) -> GroupChatState:
        return GroupChatState()

    @override
    async def handle(self) -> None:
        if self.node_state is None:
            self.node_state = GroupChatState()
        else:
            if not self.node_state.agent:
                self.node_state.agent = get_agent_app()
            if not self.node_state.storages:
                self.node_state.storages = {}

        session_id = str(self.event.group_id)
        async with self.node_state.storages_lock:
            if session_id not in self.node_state.storages:
                self.node_state.storages[session_id] = Histories()
            history_storage = self.node_state.storages[session_id]
        timestamp = datetime.fromtimestamp(self.event.time, tz=UTC)
        user = self.event.sender.nickname
        text = self.event.message.get_plain_text()
        if not text:
            return
        is_tome = self.event.is_tome()
        message = {
            "timestamp": timestamp,
            "user": user,
            "text": text,
        }

        async with history_storage.lock:
            history_storage.messages.append(message)
            history_storage.messages_for_extra_prompt.append(message)
            history_storage.messages_for_extra_prompt = (
                history_storage.messages_for_extra_prompt[-5:]
            )

            if _is_activity_limited(history_storage.activites, self.event.time):
                return

            if history_storage.on_handle:
                return

            if (
                self.event.time - history_storage.timestamp <= INTERVAL_SECONDS
                and not is_tome
                and "可不" not in text
            ):
                return

            if self.event.group_id not in ALLOWED_AUTO_REPLY_GROUPS and not is_tome:
                return

            history_storage.on_handle = True
            current_messages = history_storage.messages
            history_storage.messages = []

        replied = False
        try:
            history_text = "\n".join(
                [item["text"] for item in history_storage.messages_for_extra_prompt]
            )
            extra_prompt = get_extra_prompt(history_text)
            print(history_text, extra_prompt)
            async for reply in self.node_state.agent.astream(
                {
                    "messages": current_messages,
                    "now_time": datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "reply_format": MORE_SENTENCE_REPLY_PROMPT
                    if is_tome
                    else ONE_SENTENCE_REPLY_PROMPT,
                    "extra_prompt": extra_prompt,
                    "thinking": True,
                    "reasoning_effort": "high",
                    "is_tome": is_tome,
                },
                config={"configurable": {"session_id": session_id}},
            ):
                if reply is not None:
                    await self.reply(reply.text)
                    replied = True
        finally:
            async with history_storage.lock:
                if replied:
                    history_storage.timestamp = self.event.time
                    history_storage.activites.append(
                        datetime.fromtimestamp(self.event.time, tz=UTC)
                    )
                    if len(history_storage.activites) > ACTIVITY_HISTORY_LIMIT:
                        history_storage.activites = (
                            history_storage.activites[-ACTIVITY_HISTORY_LIMIT:]
                            if ACTIVITY_HISTORY_LIMIT > 0
                            else []
                        )
                else:
                    history_storage.messages = (
                        current_messages + history_storage.messages
                    )
                history_storage.on_handle = False
