from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
from typing_extensions import override
from zoneinfo import ZoneInfo

import anyio
from langchain_core.runnables import Runnable
from pydantic import model_validator
from sekaibot import Node
from sekaibot.adapter.cqhttp.event import GroupMessageEvent
from sekaibot.adapter.cqhttp.message import CQHTTPMessage, CQHTTPMessageSegment
from sekaibot.config import ConfigModel

from chat.activity import get_activity_store
from chat.group import (
    AssistantReply,
    get_agent_app,
)
from chat.image import search_meme
from chat.prompt import (
    MORE_SENTENCE_REPLY_PROMPT,
    ONE_SENTENCE_REPLY_PROMPT,
    get_extra_prompt,
)
from chat.utils import parse_message

BASE_AUTO_REPLY_GROUPS = {
    596488203,
    1011357049,
    1058218429,
    1087911123,
    834922207,
    895484096,
}

BACKUP_MESSAGES_LIMIT = 20

EXTRA_PROMPT_MAX_HISTORY = 5


class GroupChatConfig(ConfigModel):
    """群聊记录节点配置"""

    __config_name__ = "group_chat"

    auto_reply_groups: set[int] = BASE_AUTO_REPLY_GROUPS
    interval_seconds: int = 40
    # (window_seconds, threshold)
    activity_limits: tuple[tuple[int, int], ...] = (
        (3600 * 5, 100),
        (3600 * 24 * 7, 500),
    )

    @model_validator(mode="after")
    def _add_default_auto_reply_groups(self):
        self.auto_reply_groups = self.auto_reply_groups.union(BASE_AUTO_REPLY_GROUPS)
        return self


@dataclass
class Histories:
    messages: list[dict[str, Any]] = field(default_factory=list)
    backup_messages: list[dict[str, Any]] = field(default_factory=list)
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


class GroupChat(Node[GroupMessageEvent, GroupChatState, GroupChatConfig]):
    """群聊记录节点"""

    priority = 1

    @override
    def __init_state__(self) -> GroupChatState:
        return GroupChatState()

    async def _is_activity_limited(self, session_id: str, event_time: int) -> bool:
        return await get_activity_store().is_limited(
            scope="group",
            session_id=session_id,
            event_time=event_time,
            activity_limits=self.config.activity_limits,
        )

    async def _get_history_storage(
        self,
        session_id: str,
    ) -> Histories:
        async with self.node_state.storages_lock:
            if session_id not in self.node_state.storages:
                self.node_state.storages[session_id] = Histories()
            return self.node_state.storages[session_id]

    async def _claim_messages_for_reply(
        self,
        session_id: str,
        history_storage: Histories,
        message: dict[str, Any],
        text: str,
        is_tome: bool,
    ) -> list[dict[str, Any]] | None:
        async with history_storage.lock:
            history_storage.messages.append(message)
            history_storage.backup_messages.append(message)
            history_storage.backup_messages = history_storage.backup_messages[
                -BACKUP_MESSAGES_LIMIT:
            ]

            if session_id not in BASE_AUTO_REPLY_GROUPS and (
                await self._is_activity_limited(session_id, self.event.time)
            ):
                return None

            if history_storage.on_handle:
                return None

            if (
                self.event.time - history_storage.timestamp
                <= self.config.interval_seconds
                and not is_tome
                and "可不" not in text
            ):
                return None

            if self.event.group_id not in self.config.auto_reply_groups and not is_tome:
                return None

            history_storage.on_handle = True
            current_messages = history_storage.messages
            history_storage.messages = []
            return current_messages

    async def _run_agent_reply(
        self,
        session_id: str,
        history_storage: Histories,
        current_messages: list[dict[str, Any]],
        is_tome: bool,
    ) -> bool:
        replied = False
        history_text = "\n".join(
            [
                item["text"]
                for item in history_storage.backup_messages[-EXTRA_PROMPT_MAX_HISTORY:]
            ]
        )
        extra_prompt = get_extra_prompt(history_text)
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
                print(f"Reply-Group: {reply.text}")
                segments, text = parse_message(reply.text.strip())
                message: CQHTTPMessage | str = ""
                for seg in segments:
                    if seg.type == "at" and "name" in seg.data:
                        if name := seg.data["name"]:
                            for item in current_messages:
                                if item["user"] == name:
                                    user_id = int(item["user_id"])
                                    message += CQHTTPMessageSegment.at(user_id) + " "
                                    break
                    elif seg.type == "reply" and "time" in seg.data:
                        have_reply = False
                        if isinstance(message, CQHTTPMessage):
                            for item in message:
                                if (
                                    isinstance(item, CQHTTPMessageSegment)
                                    and item.type == "reply"
                                ):
                                    have_reply = True
                                    break
                        if not have_reply and (time := seg.data["time"]):
                            for item in current_messages:
                                t = cast("datetime", item["timestamp"]).astimezone(
                                    ZoneInfo("Asia/Shanghai")
                                )
                                if t.strftime("%H:%M:%S") in time:
                                    message_id = int(item["message_id"])
                                    message += CQHTTPMessageSegment.reply(message_id)
                                    break
                    elif seg.type == "meme" and "content" in seg.data:
                        content = seg.data["content"]
                        meme_result = await search_meme(
                            content, temperature=0, max_score=1.5
                        )
                        if meme_result:
                            message += CQHTTPMessageSegment.image(
                                meme_result.base64, sub_type=1
                            )
                            await self.reply(message)
                            message = ""
                    else:
                        continue
                message += text
                if message:
                    await self.reply(message)
                replied = True
        return replied

    async def _finish_reply_attempt(
        self,
        history_storage: Histories,
        current_messages: list[dict[str, Any]],
        replied: bool,
    ) -> None:
        async with history_storage.lock:
            try:
                if replied:
                    history_storage.timestamp = self.event.time
                    await get_activity_store().record(
                        scope="group",
                        session_id=str(self.event.group_id),
                        event_time=self.event.time,
                        weight=1.0,
                        activity_limits=self.config.activity_limits,
                    )
                else:
                    history_storage.messages = (
                        current_messages + history_storage.messages
                    )
            finally:
                history_storage.on_handle = False

    @override
    async def handle(self) -> None:
        session_id = str(self.event.group_id)
        history_storage = await self._get_history_storage(session_id)

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
            "user_id": self.event.sender.user_id,
            "message_id": self.event.message_id,
        }

        current_messages = await self._claim_messages_for_reply(
            session_id=session_id,
            history_storage=history_storage,
            message=message,
            text=text,
            is_tome=is_tome,
        )
        if current_messages is None:
            return

        replied = False
        try:
            replied = await self._run_agent_reply(
                session_id=session_id,
                history_storage=history_storage,
                current_messages=current_messages,
                is_tome=is_tome,
            )
        finally:
            await self._finish_reply_attempt(
                history_storage=history_storage,
                current_messages=current_messages,
                replied=replied,
            )
