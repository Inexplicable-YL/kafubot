import json
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
    clear_session_history,
    get_chat_app,
    get_decision_app,
)
from chat.image import (
    ImageReadResult,
    get_image_analyzer,
    read_image,
)
from chat.meme import add_memes, search_meme
from chat.prompt import get_extra_prompt
from chat.utils import parse_message

MESSAGES_LIMIT = 30
BACKUP_MESSAGES_LIMIT = 20
BACKUP_PENDING_COUNTS_LIMIT = 10

EXTRA_PROMPT_MAX_HISTORY = 5

DEFAULT_USERNAME = "陌生用户"

DEFAULT_CLEAR_KEYWORDS = {"/clear", "/清除"}


def _ttest_signal(statistic: Any) -> float:
    value = float(statistic)
    if math.isnan(value):
        return 0.0
    return -(math.tanh(value) if value > 0 else value)


class GroupChatConfig(ConfigModel):
    __config_name__ = "group_chat"

    unrestricted_groups: set[int] = set()
    auto_reply_groups: set[int] = set()
    interval_seconds: int = 3
    keep_image_limit: int = 3
    talk_value: float = 0.8
    reply_keywords: set[str] = set()
    reply_when_keywords: bool = False
    clear_keywords: set[str] = set()
    # (window_seconds, threshold)
    activity_limits: tuple[tuple[int, int], ...] = (
        (3600 * 5, 100),
        (3600 * 24 * 7, 500),
    )

    @model_validator(mode="after")
    def _add_default(self):
        self.auto_reply_groups = self.auto_reply_groups.union(self.unrestricted_groups)
        self.clear_keywords = self.clear_keywords.union(DEFAULT_CLEAR_KEYWORDS)
        return self


class GroupMessage(BaseModel):
    role: Literal["user", "assistant"] = "user"
    timestamp: datetime
    user: str
    text: str
    user_id: str
    message_id: str | None = None
    is_tome: bool
    to_other: bool
    have_keywords: bool


class GroupImageMessage(GroupMessage):
    image: ImageReadResult
    as_meme: bool


class Histories(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: deque[GroupMessage] = Field(
        default_factory=lambda: deque(maxlen=MESSAGES_LIMIT)
    )
    backup_messages: deque[GroupMessage] = Field(
        default_factory=lambda: deque(maxlen=BACKUP_MESSAGES_LIMIT)
    )
    backup_pending_counts: deque[int] = Field(
        default_factory=lambda: deque(maxlen=BACKUP_PENDING_COUNTS_LIMIT)
    )
    on_handle: bool = False
    handle_condition: anyio.Condition = Field(default_factory=anyio.Condition)
    lock: anyio.Lock = Field(default_factory=anyio.Lock)


class GroupEvent(BaseModel):
    session_id: str
    history_storage: Histories
    message: GroupMessage

    @property
    def time(self) -> int:
        return int(self.message.timestamp.timestamp())


class GroupChatState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    decision: Runnable[dict[str, Any], bool] = Field(default_factory=get_decision_app)
    chat: Runnable[dict[str, Any], str] = Field(default_factory=get_chat_app)
    image_analyzer: Runnable[dict[str, Any], str] = Field(
        default_factory=lambda _: get_image_analyzer(True, add_memes_hook=add_memes)
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
        group_event: GroupEvent,
    ) -> bool:
        session_id = group_event.session_id
        history_storage = group_event.history_storage
        event_time = group_event.time
        if (
            int(session_id) not in self.config.unrestricted_groups
            and await self.node_state.activity_store.is_limited(
                scope="group",
                session_id=session_id,
                event_time=event_time,
                activity_limits=self.config.activity_limits,
            )
        ):
            return True

        if latest_timestamp := await self.node_state.activity_store.latest_timestamp(
            scope="group", session_id=session_id
        ):
            if event_time - latest_timestamp <= self.config.interval_seconds:
                return True
            if recent_interval := await self.node_state.activity_store.recent_interval(
                scope="group", session_id=session_id, window_seconds=600
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
                    7
                    * math.tanh((pending + interval) / 4)
                    / (8 * self.config.talk_value)
                )

                return equivalent_pending <= 1 / self.config.talk_value
            return False
        return False

    async def claim_messages(  # noqa: PLR0911
        self,
        group_event: GroupEvent,
    ) -> list[GroupMessage] | None:
        history_storage = group_event.history_storage
        message = group_event.message
        current_messages: list[GroupMessage] | None = None
        is_tome = message.is_tome or any(
            msg.is_tome for msg in history_storage.messages
        )
        await history_storage.lock.acquire()
        is_released = False
        try:
            history_storage.messages.append(message)
            len_messages = len(history_storage.messages)
            history_storage.backup_messages.append(message)
            if isinstance(group_event.message, GroupImageMessage):
                return None
            if self.event.group_id not in self.config.auto_reply_groups and not is_tome:
                return None

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
                if group_event.message.to_other:
                    return None
                # Activity is limited, skip.
                if await self.is_activity_limited(group_event):
                    return None
            history_storage.on_handle = True
            current_messages = list(history_storage.messages)
            history_storage.messages.clear()
        finally:
            if not is_released:
                history_storage.lock.release()
        return current_messages

    async def filling_images(
        self,
        current_messages: list[GroupMessage],
    ) -> list[GroupMessage]:
        if not current_messages:
            return []
        skip = (
            sum(isinstance(x, GroupImageMessage) for x in current_messages)
            - self.config.keep_image_limit
        )
        messages = [
            x
            for x in current_messages
            if not isinstance(x, GroupImageMessage) or (skip := skip - 1) < 0
        ]
        image_inputs = [
            {
                "image": x.image.base64,
                "phash": x.image.phash,
                "as_meme": x.as_meme,
                "detail": False,
            }
            for x in messages
            if isinstance(x, GroupImageMessage) and not x.text.strip()
        ]
        if not image_inputs:
            return messages
        results = await self.node_state.image_analyzer.abatch(image_inputs)
        filled_messages = []
        for msg in messages:
            if isinstance(msg, GroupImageMessage) and not msg.text.strip():
                msg.text = results.pop(0)
            filled_messages.append(msg)
        return filled_messages

    async def run_reply(
        self,
        group_event: GroupEvent,
        *,
        current_messages: list[GroupMessage],
    ) -> list[GroupMessage] | None:
        fill_event = anyio.Event()
        output_messages: list[GroupMessage] | None = current_messages

        async def _fill(fill_event: anyio.Event):
            nonlocal current_messages
            try:
                current_messages = await self.filling_images(current_messages)
            finally:
                fill_event.set()

        async def _reply(fill_event: anyio.Event) -> list[GroupMessage] | None:
            nonlocal output_messages
            should_reply = (
                group_event.message.is_tome
                or any(msg.is_tome for msg in group_event.history_storage.messages)
                or await self.node_state.decision.ainvoke(
                    {
                        "messages": [
                            item.model_dump(include={"timestamp", "user", "text"})
                            for item in current_messages
                        ],
                    },
                    config={"configurable": {"session_id": group_event.session_id}},
                )
            )
            await fill_event.wait()
            output_messages = current_messages
            if should_reply:
                print(f"Invoking-Group: {[m.text for m in current_messages]}")
                output_messages = await self.get_reply(
                    group_event,
                    current_messages=current_messages,
                )

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(_fill, fill_event)
            task_group.start_soon(_reply, fill_event)

        return output_messages

    async def get_reply(
        self,
        group_event: GroupEvent,
        *,
        current_messages: list[GroupMessage],
    ) -> list[GroupMessage] | None:
        history_storage = group_event.history_storage
        history_text = "\n".join(
            [
                item.text
                for item in list(history_storage.backup_messages)[
                    -EXTRA_PROMPT_MAX_HISTORY:
                ]
            ]
        )
        extra_prompt = await get_extra_prompt(history_text)
        message: CQHTTPMessage | str = ""
        full_text: str = ""
        async for reply in self.node_state.chat.astream(
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
            },
            config={"configurable": {"session_id": group_event.session_id}},
        ):
            if reply is not None and (reply_msg := reply.strip()):
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
                        meme_result = await search_meme(
                            content, temperature=0.5, min_score=0.0
                        )
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
                    is_tome=False,
                    to_other=False,
                    have_keywords=False,
                )
            )
            history_storage.backup_pending_counts.append(len(current_messages))
            return None
        return current_messages

    async def finish_reply(
        self,
        group_event: GroupEvent,
        *,
        current_messages: list[GroupMessage] | None = None,
    ) -> None:
        history_storage = group_event.history_storage
        async with history_storage.lock:
            if current_messages:
                history_storage.messages.extendleft(current_messages)
            history_storage.on_handle = False
            async with history_storage.handle_condition:
                history_storage.handle_condition.notify_all()
        if current_messages is None:
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

    async def get_message_or_clear_chat(
        self, session_id: str, history_storage: Histories, to_me: bool
    ) -> tuple[tuple[ImageReadResult, bool] | str | None, bool]:
        text = self.event.message.get_plain_text().strip()
        if text:
            if any(keyw in text for keyw in self.config.clear_keywords):
                await self.delete_chat(session_id=session_id)
                return None, False
        elif (
            len(self.event.message) == 1
            and self.event.message[0].type == "image"
            and (file := self.event.message[0].data.get("file"))
        ):
            as_meme = str(self.event.message[0].data.get("sub_type", "0")) == "1"
            if image := await self.get_image(file):
                return (image, as_meme), False
            return None, False
        elif (
            len(self.event.message) == 1
            and self.event.message[0].type == "file"
            and (file_name := self.event.message[0].data.get("file"))
        ):
            text = f"[文件:{file_name}]"
        elif (
            len(self.event.message) == 1
            and self.event.message[0].type == "json"
            and (
                prompt := json.loads(self.event.message[0].data.get("data", "{}")).get(
                    "prompt"
                )
            )
        ):
            text = f"[小程序:{prompt}]"
        else:
            return None, False

        to_other = False
        if to_me:
            text = f"[MSG:at, name=可不] {text}"
        for msg in self.event.message:
            if (
                isinstance(msg, CQHTTPMessageSegment)
                and msg.type == "at"
                and (qq_number := str(msg.data.get("qq", "")))
            ):
                at_name: str | None = None
                for item in history_storage.backup_messages:
                    if str(item.user_id) == qq_number:
                        at_name = item.user
                        break
                if at_name:
                    text = f"[MSG:at, name={at_name}] {text}"
                to_other = not to_me

        if self.event.reply and (reply_time := int(self.event.reply.time)):
            time_text = (
                datetime.fromtimestamp(reply_time, tz=UTC)
                .astimezone(ZoneInfo("Asia/Shanghai"))
                .strftime("%Y-%m-%d %H:%M:%S")
            )
            text = f"[MSG:reply, time={time_text}] {text}"
            to_other = to_other or (
                self.event.reply.sender.user_id != self.event.adapter.self_id
            )
        return text, to_other

    def have_keywords(self, text: str) -> bool:
        text = (
            opencc.OpenCC("s2t").convert(text)
            + "\n"
            + opencc.OpenCC("t2s").convert(text)
        )
        return any(keyw in text for keyw in self.config.reply_keywords)

    async def get_image(self, file: str) -> ImageReadResult | None:
        try:
            result: dict[str, str] = await self.event.adapter.call_api(
                "get_image", file=file
            )
            path, url = (
                result.get("file"),
                result.get("url"),
            )
            if path is not None:
                return await read_image(path, url)
        except Exception:
            return None
        return None

    async def get_event(self) -> GroupEvent | None:
        session_id = str(self.event.group_id)
        history_storage = await self.get_history_storage(session_id)
        is_tome = self.event.is_tome()
        message, to_other = await self.get_message_or_clear_chat(
            session_id, history_storage, is_tome
        )
        if message is None:
            return None
        timestamp = datetime.fromtimestamp(self.event.time, tz=UTC)
        user = self.event.sender.nickname or DEFAULT_USERNAME
        have_keywords = False
        if isinstance(message, tuple):
            message = GroupImageMessage(
                role="user",
                timestamp=timestamp,
                user=user,
                text="",
                image=message[0],
                as_meme=message[1],
                user_id=str(self.event.user_id),
                message_id=str(self.event.message_id),
                is_tome=is_tome,
                have_keywords=have_keywords,
                to_other=to_other,
            )
        else:
            have_keywords = self.have_keywords(message)
            if self.config.reply_when_keywords:
                is_tome = is_tome or have_keywords
            message = GroupMessage(
                role="user",
                timestamp=timestamp,
                user=user,
                text=message,
                user_id=str(self.event.user_id),
                message_id=str(self.event.message_id),
                is_tome=is_tome,
                have_keywords=have_keywords,
                to_other=to_other,
            )
        return GroupEvent(
            session_id=session_id,
            history_storage=history_storage,
            message=message,
        )

    @override
    async def handle(self) -> None:
        group_event = await self.get_event()
        if group_event is None:
            return
        current_messages = await self.claim_messages(group_event)
        if current_messages is None:
            return

        try:
            current_messages = await self.run_reply(
                group_event,
                current_messages=current_messages,
            )
        finally:
            await self.finish_reply(
                group_event,
                current_messages=current_messages,
            )

    @override
    async def rule(self) -> bool:
        return str(self.event.user_id) != "2830758180"
