import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast
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
from sekaibot.adapter.cqhttp.message import CQHTTPMessageSegment

INTERVAL_SECONDS = 40
ACTIVITY_LIMITS: tuple[tuple[int, int], ...] = (  # (window_seconds, threshold)
    (300, 20),
    (3600, 40),
    (3600 * 5, 100),
    (3600 * 24 * 7, 500),
)
ACTIVITY_HISTORY_LIMIT = max((threshold for _, threshold in ACTIVITY_LIMITS), default=0)

ALLOWED_AUTO_REPLY_GROUPS = {
    596488203,
    1011357049,
    1058218429,
    1087911123,
    834922207,
    648749016,
}

BACKUP_MESSAGES_LIMIT = 20

EXTRA_PROMPT_MAX_HISTORY = 5


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
    backup_messages: list[dict[str, Any]] = field(default_factory=list)
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


def extract(text: str) -> tuple[str | None, str | None, str]:
    at_pattern = re.compile(r"\[MSG:at\b[^\]]*?\bname=([^,\]]+)")
    reply_pattern = re.compile(r"\[MSG:reply\b[^\]]*?\btime=([^,\]]+)")
    at_remove = re.compile(r"\[MSG:at\b[^\]]*\]")
    reply_remove = re.compile(r"\[MSG:reply\b[^\]]*\]")
    name = None
    at_match = at_pattern.search(text)
    if at_match:
        name = str(at_match.group(1)).strip()
    time = None
    reply_match = reply_pattern.search(text)
    if reply_match:
        time = str(reply_match.group(1)).strip()
    clean_text = at_remove.sub("", text)
    clean_text = reply_remove.sub("", clean_text)
    return name, time, clean_text.strip()


class GroupChat(Node[GroupMessageEvent, GroupChatState, Any]):
    """群聊记录节点"""

    priority = 1

    @override
    def __init_state__(self) -> GroupChatState:
        return GroupChatState()

    def _ensure_state(self) -> None:
        if self.node_state is None:
            self.node_state = GroupChatState()
        if not self.node_state.agent:
            self.node_state.agent = get_agent_app()
        if not self.node_state.storages:
            self.node_state.storages = {}

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

            if _is_activity_limited(history_storage.activites, self.event.time):
                return None

            if history_storage.on_handle:
                return None

            if (
                self.event.time - history_storage.timestamp <= INTERVAL_SECONDS
                and not is_tome
                and "可不" not in text
            ):
                return None

            if self.event.group_id not in ALLOWED_AUTO_REPLY_GROUPS and not is_tome:
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
                print(f"Agent reply: {reply.text}")
                name, time, text = extract(reply.text)
                user_id: int | None = None
                message_id: int | None = None
                if name:
                    for item in current_messages:
                        if item["user"] == name:
                            user_id = int(item["user_id"])
                            break
                if time:
                    for item in current_messages:
                        t = cast("datetime", item["timestamp"]).astimezone(
                            ZoneInfo("Asia/Shanghai")
                        )
                        if t.strftime("%H:%M:%S") in time:
                            message_id = int(item["message_id"])
                            break
                message = ""
                if user_id:
                    message += CQHTTPMessageSegment.at(user_id) + " "
                if message_id:
                    message += CQHTTPMessageSegment.reply(message_id)
                await self.reply(message + text)
                replied = True
        return replied

    async def _finish_reply_attempt(
        self,
        history_storage: Histories,
        current_messages: list[dict[str, Any]],
        replied: bool,
    ) -> None:
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
                history_storage.messages = current_messages + history_storage.messages
            history_storage.on_handle = False

    @override
    async def handle(self) -> None:
        self._ensure_state()
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
