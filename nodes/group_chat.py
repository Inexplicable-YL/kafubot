import math
from collections import deque
from datetime import UTC, datetime
from typing import Any, Literal, cast
from typing_extensions import override
from zoneinfo import ZoneInfo

import anyio
import opencc
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats  # type: ignore[import-untyped]
from sekaibot import Node
from sekaibot.adapter.cqhttp.event import GroupMessageEvent
from sekaibot.adapter.cqhttp.message import CQHTTPMessage, CQHTTPMessageSegment
from sekaibot.config import ConfigModel

from chat.activity import ActivityStore, get_activity_store
from chat.group import (
    AssistantReply,
    clear_session_history,
    get_agent_app,
)
from chat.image import search_meme
from chat.prompt import (
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

BACKUP_MESSAGES_LIMIT = 40
BACKUP_PENDING_COUNTS_LIMIT = 10

EXTRA_PROMPT_MAX_HISTORY = 5

DEFAULT_USERNAME = "用户"

DELETE_MESSAGE_KEYWORDS = ["/clear", "/清除"]


def _ttest_signal(statistic: Any) -> float:
    value = float(statistic)
    if math.isnan(value):
        return 0.0
    return -(math.tanh(value) if value > 0 else value)


class GroupChatConfig(ConfigModel):
    """群聊记录节点配置"""

    __config_name__ = "group_chat"

    auto_reply_groups: set[int] = BASE_AUTO_REPLY_GROUPS
    interval_seconds: int = 40
    talk_value: float = 0.8
    keywords: set[str] = {"可不", "花谱"}
    reply_when_keywords: bool = False
    # (window_seconds, threshold)
    activity_limits: tuple[tuple[int, int], ...] = (
        (3600 * 5, 100),
        (3600 * 24 * 7, 500),
    )

    @model_validator(mode="after")
    def _add_default_auto_reply_groups(self):
        self.auto_reply_groups = self.auto_reply_groups.union(BASE_AUTO_REPLY_GROUPS)
        return self


class GroupMessage(BaseModel):
    role: Literal["user", "assistant"] = "user"
    timestamp: datetime
    user: str
    text: str
    user_id: str
    message_id: str | None = None


class Histories(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[GroupMessage] = Field(default_factory=list)
    backup_messages: deque[GroupMessage] = Field(
        default_factory=lambda: deque(maxlen=BACKUP_MESSAGES_LIMIT)
    )
    backup_pending_counts: deque[int] = Field(
        default_factory=lambda: deque(maxlen=BACKUP_PENDING_COUNTS_LIMIT)
    )
    on_handle: bool = False
    handle_condition: anyio.Condition = Field(default_factory=anyio.Condition)
    lock: anyio.Lock = Field(default_factory=anyio.Lock)

    @property
    def latest_timestamp(self) -> int:
        if self.backup_messages:
            return int(self.backup_messages[-1].timestamp.timestamp())
        return 0


class GroupChatState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Runnable[dict[str, Any], None | AssistantReply] = Field(
        default_factory=get_agent_app
    )
    activity_store: ActivityStore = Field(default_factory=get_activity_store)
    storages: dict[str, Histories] = Field(default_factory=dict)
    storages_lock: anyio.Lock = Field(default_factory=anyio.Lock)


class GroupChat(Node[GroupMessageEvent, GroupChatState, GroupChatConfig]):
    """群聊记录节点"""

    priority = 1

    @override
    def __init_state__(self) -> GroupChatState:
        return GroupChatState()

    async def get_history_storage(
        self,
        session_id: str,
    ) -> Histories:
        async with self.node_state.storages_lock:
            if session_id not in self.node_state.storages:
                self.node_state.storages[session_id] = Histories()
            return self.node_state.storages[session_id]

    async def is_activity_limited(
        self,
        session_id: str,
        event_time: int,
        history_storage: Histories,
    ) -> bool:
        if int(session_id) in BASE_AUTO_REPLY_GROUPS:
            return False
        if await self.node_state.activity_store.is_limited(
            scope="group",
            session_id=session_id,
            event_time=event_time,
            activity_limits=self.config.activity_limits,
        ):
            return True
        if (
            latest_timestamp := await self.node_state.activity_store.latest_timestamp(
                scope="group", session_id=session_id
            )
        ) and (
            recent_interval := await self.node_state.activity_store.recent_interval(
                scope="group", session_id=session_id, window_seconds=600
            )
        ):
            if len(history_storage.backup_pending_counts) > 1:
                pending_statistic, _ = cast(
                    "tuple[Any, Any]",
                    stats.ttest_1samp(
                        history_storage.backup_pending_counts,
                        popmean=len(history_storage.messages),
                    ),
                )
                pending = _ttest_signal(pending_statistic)
            else:
                pending = 2.0
            if len(recent_interval) > 1:
                interval_statistic, _ = cast(
                    "tuple[Any, Any]",
                    stats.ttest_1samp(
                        recent_interval,
                        popmean=event_time - latest_timestamp,
                    ),
                )
                interval = _ttest_signal(interval_statistic)
            else:
                interval = 2.0
            equivalent_pending = len(history_storage.messages) + (
                7 * math.tanh((pending + interval) / 4) / (8 * self.config.talk_value)
            )

            return equivalent_pending <= 1 / self.config.talk_value
        return False

    async def claim_messages(  # noqa: PLR0911
        self,
        session_id: str,
        history_storage: Histories,
        message: GroupMessage,
        is_tome: bool,
        have_keywords: bool,
        to_other: bool,
    ) -> list[GroupMessage] | None:
        await history_storage.lock.acquire()
        is_released = False
        try:
            history_storage.messages.append(message)
            len_messages = len(history_storage.messages)
            latest_timestamp = history_storage.latest_timestamp
            history_storage.backup_messages.append(message)

            if history_storage.on_handle:
                history_storage.lock.release()
                is_released = True
                with anyio.move_on_after(delay=30) as scope:
                    async with history_storage.handle_condition:
                        await history_storage.handle_condition.wait_for(
                            lambda: not history_storage.on_handle
                        )
                # Waiting for the previous event to time out, skip.
                if scope.cancelled_caught:
                    return None
                await history_storage.lock.acquire()
                is_released = False
                # Another waiter may already have claimed the pending messages.
                if history_storage.on_handle:
                    return None
                # Further events occurred after this one, skip.
                if not history_storage.messages or history_storage.messages[
                    -1
                ].message_id != str(self.event.message_id):
                    return None

            # There was no response to the previous incident and the message did not refer to the bot.
            if len_messages == len(history_storage.messages) and not is_tome:
                # The message is a reply to someone else, skip.
                if to_other:
                    return None
                # Group chat is not configured to automatically reply, skip.
                if self.event.group_id not in self.config.auto_reply_groups:
                    return None
                # The message intervals were too short and no keywords were included, skip.
                if (
                    self.event.time - latest_timestamp <= self.config.interval_seconds
                    and not have_keywords
                ):
                    return None
                # Activity is limited, skip.
                if await self.is_activity_limited(
                    session_id, self.event.time, history_storage
                ):
                    return None
            history_storage.on_handle = True
            current_messages = history_storage.messages
            history_storage.messages = []
            return current_messages
        finally:
            if not is_released:
                history_storage.lock.release()

    async def run_reply(
        self,
        session_id: str,
        history_storage: Histories,
        current_messages: list[GroupMessage],
        is_tome: bool,
    ) -> bool:
        history_text = "\n".join(
            [
                item.text
                for item in list(history_storage.backup_messages)[
                    -EXTRA_PROMPT_MAX_HISTORY:
                ]
            ]
        )
        extra_prompt = get_extra_prompt(history_text)

        message: CQHTTPMessage | str = ""
        full_text: str = ""
        async for reply in self.node_state.agent.astream(
            {
                "messages": [
                    item.model_dump(include={"timestamp", "user", "text"})
                    for item in current_messages
                ],
                "now_time": datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "extra_prompt": extra_prompt,
                "thinking": True,
                "reasoning_effort": "high",
                "is_tome": is_tome,
            },
            config={"configurable": {"session_id": session_id}},
        ):
            if reply is not None and (reply_msg := reply.text.strip()):
                print(f"Reply-Group: {reply_msg}")
                full_text += reply_msg + "\n"
                segments, text = parse_message(reply_msg)
                for seg in segments:
                    if seg.type == "at" and "name" in seg.data:
                        if name := seg.data["name"]:
                            for item in current_messages:
                                if item.user == name:
                                    user_id = int(item.user_id)
                                    message += CQHTTPMessageSegment.at(user_id) + " "
                                    break
                    elif seg.type == "reply" and "time" in seg.data:
                        have_reply = False
                        if isinstance(message, CQHTTPMessage):
                            for message_segment in message:
                                if (
                                    isinstance(message_segment, CQHTTPMessageSegment)
                                    and message_segment.type == "reply"
                                ):
                                    have_reply = True
                                    break
                        if not have_reply and (time := seg.data["time"]):
                            for item in current_messages:
                                t = cast("datetime", item.timestamp).astimezone(
                                    ZoneInfo("Asia/Shanghai")
                                )
                                if t.strftime("%H:%M:%S") in time and (
                                    message_id := item.message_id
                                ):
                                    message += CQHTTPMessageSegment.reply(
                                        int(message_id)
                                    )
                                    break
                    elif seg.type == "meme" and "content" in seg.data:
                        content = seg.data["content"]
                        meme_result = await search_meme(content, temperature=0.15, min_score=0.05)
                        if meme_result:
                            await self.reply(
                                CQHTTPMessageSegment.image(
                                    meme_result.base64, sub_type=1
                                )
                            )
                        message = ""
                    else:
                        continue
                message += text
                if (isinstance(message, str) and message.strip()) or (
                    isinstance(message, CQHTTPMessage)
                    and message.get_plain_text().strip()
                ):
                    await self.reply(message)
                    message = ""
        if full_text := full_text.strip():
            history_storage.backup_messages.append(
                GroupMessage(
                    role="assistant",
                    timestamp=datetime.now(tz=UTC),
                    user="可不",
                    text=full_text,
                    user_id=str(self.event.adapter.self_id),
                    message_id="",
                )
            )
            history_storage.backup_pending_counts.append(len(current_messages))
            return True
        return False

    async def finish_reply(
        self,
        history_storage: Histories,
        current_messages: list[GroupMessage],
        replied: bool,
    ) -> None:
        async with history_storage.lock:
            if not replied:
                history_storage.messages = current_messages + history_storage.messages
            history_storage.on_handle = False
            async with history_storage.handle_condition:
                history_storage.handle_condition.notify_all()
        if replied:
            await self.node_state.activity_store.record(
                scope="group",
                session_id=str(self.event.group_id),
                event_time=self.event.time,
                weight=1.0,
                activity_limits=self.config.activity_limits,
            )

    async def delete_chat(self, session_id: str) -> None:
        await clear_session_history(session_id=session_id)
        async with self.node_state.storages_lock:
            if session_id in self.node_state.storages:
                del self.node_state.storages[session_id]
        await self.reply("[SYSTEM]已清除历史", at_sender=True)

    async def get_text(
        self, session_id: str, history_storage: Histories, to_me: bool
    ) -> tuple[str | None, bool]:
        text = self.event.message.get_plain_text().strip()
        if not text:
            return None, False
        if any(keyw in text for keyw in DELETE_MESSAGE_KEYWORDS):
            await self.delete_chat(session_id=session_id)
            return None, False
        to_other = False
        for msg in self.event.message:
            if (
                isinstance(msg, CQHTTPMessageSegment)
                and msg.type == "at"
                and (qq_number := str(msg.data.get("qq", "")))
            ):
                if qq_number == str(self.event.adapter.self_id):
                    text = f"[MSG:at, name=可不]{text}"
                    continue
                at_name: str | None = None
                for item in history_storage.backup_messages:
                    if str(item.user_id) == qq_number:
                        at_name = item.user
                        break
                if at_name:
                    text = f"[MSG:at, name={at_name}]{text}"
                to_other = not to_me
        if self.event.reply and (reply_id := str(self.event.reply.message_id)):
            reply_time: str | None = None
            for item in history_storage.backup_messages:
                if str(item.message_id) == reply_id:
                    t = cast("datetime", item.timestamp).astimezone(
                        ZoneInfo("Asia/Shanghai")
                    )
                    reply_time = t.strftime("%Y-%m-%d %H:%M:%S")
                    break
            if reply_time:
                text = f"[MSG:reply, time={reply_time}]{text}"
            to_other = not to_me
        return text, to_other

    def have_keywords(self, text: str) -> bool:
        text = (
            opencc.OpenCC("s2t").convert(text)
            + "\n"
            + opencc.OpenCC("t2s").convert(text)
        )
        return any(keyw in text for keyw in self.config.keywords)

    @override
    async def handle(self) -> None:
        session_id = str(self.event.group_id)
        history_storage = await self.get_history_storage(session_id)
        is_tome = self.event.is_tome()
        text, to_other = await self.get_text(session_id, history_storage, is_tome)
        if not text:
            return
        timestamp = datetime.fromtimestamp(self.event.time, tz=UTC)
        user = self.event.sender.nickname or DEFAULT_USERNAME
        have_keywords = self.have_keywords(text)
        if self.config.reply_when_keywords:
            is_tome = is_tome or have_keywords

        message = GroupMessage(
            role="user",
            timestamp=timestamp,
            user=user,
            text=text,
            user_id=str(self.event.user_id),
            message_id=str(self.event.message_id),
        )

        current_messages = await self.claim_messages(
            session_id=session_id,
            history_storage=history_storage,
            message=message,
            is_tome=is_tome,
            have_keywords=have_keywords,
            to_other=to_other,
        )
        if current_messages is None:
            return

        replied = False
        try:
            replied = await self.run_reply(
                session_id=session_id,
                history_storage=history_storage,
                current_messages=current_messages,
                is_tome=is_tome,
            )
        finally:
            await self.finish_reply(
                history_storage=history_storage,
                current_messages=current_messages,
                replied=replied,
            )

    @override
    async def rule(self) -> bool:
        return str(self.event.user_id) != "2830758180"
