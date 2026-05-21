from collections import deque
from datetime import UTC, datetime
from typing import Any, cast
from typing_extensions import override
from zoneinfo import ZoneInfo

import anyio
from anyio.abc import TaskGroup
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sekaibot import Node
from sekaibot.adapter.cqhttp.event import PrivateMessageEvent
from sekaibot.config import ConfigModel

from chat.activity import ActivityStore, get_activity_store
from chat.image import (
    ImageReadResult,
    get_image_analyzer,
    read_image,
)
from chat.meme import add_memes
from chat.message import QQMessage, QQMessageSegment
from chat.private import (
    IMAGE_SEGMENT_TYPES,
    PrivateMessage,
    clear_session_history,
    get_chat_app,
    get_session_history,
)
from chat.prompt import get_extra_prompt

BACKUP_MESSAGES_LIMIT = 10
EXTRA_PROMPT_MAX_HISTORY = 3


DEFAULT_CLEAR_KEYWORDS = {"/clear", "/清除"}


class PrivateChatConfig(ConfigModel):
    """私聊记录节点配置"""

    __config_name__ = "private_chat"

    merge_window_seconds: float = 5
    image_analyzer_workers: int = 10
    image_hash_similarity_threshold: int = 5
    clear_keywords: set[str] = Field(
        default_factory=lambda: DEFAULT_CLEAR_KEYWORDS.copy()
    )
    activity_day: tuple[int, int] = (8, 20)
    activity_day_multiplier: float = 0.5
    activity_night_multiplier: float = 1.0
    # (window_seconds, threshold)
    activity_limits: tuple[tuple[int, int], ...] = (
        (3600 * 5, 100),
        (3600 * 24 * 7, 450),
    )

    @model_validator(mode="after")
    def _validate_activity_limits(self):
        if any(
            threshold < 0 or window_seconds < 0
            for window_seconds, threshold in self.activity_limits
        ):
            raise ValueError(
                "activity limits must have non-negative window and threshold"
            )
        if self.image_analyzer_workers < 1:
            raise ValueError("image_analyzer_workers must be positive")
        if not 0 <= self.image_hash_similarity_threshold <= 64:  # noqa: PLR2004
            raise ValueError("image_hash_similarity_threshold must be between 0 and 64")
        return self


class PrivateEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str
    message: PrivateMessage

    @property
    def time(self) -> int:
        return int(self.message.timestamp.timestamp())


class SessionQueueState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    running: bool = False
    pending_messages: list[PrivateMessage] = Field(default_factory=list)
    merge_window_seconds: float = 5
    pending_image_ids: set[str] = Field(default_factory=set)
    dropped_message_ids: set[str] = Field(default_factory=set)
    condition: anyio.Condition = Field(default_factory=anyio.Condition)

    async def enqueue(
        self,
        message: PrivateMessage,
        similarity_threshold: int,
    ) -> PrivateMessage | None:
        if not message.message:
            return None
        image_message = message if message.images else None
        async with self.condition:
            if image_message is not None:
                image_hash = image_message.images[0][0].phash
                if any(
                    item.images
                    and item.message_id not in self.dropped_message_ids
                    and item.images
                    and item.images[0][0].phash - image_hash <= similarity_threshold
                    for item in self.pending_messages
                ):
                    return None
                self.pending_image_ids.add(image_message.message_id)
            self.pending_messages.append(message)
            self.condition.notify_all()
            return image_message

    async def finish_image(
        self,
        message: PrivateMessage,
        abstract: str | None,
    ) -> None:
        async with self.condition:
            self.pending_image_ids.discard(message.message_id)
            if abstract or message.has_model_visible_content():
                message.fill_first_pending_image(abstract)
                if not message.message:
                    self.dropped_message_ids.add(message.message_id)
            else:
                self.dropped_message_ids.add(message.message_id)
            self.condition.notify_all()

    async def wait_and_claim(self, message_id: str) -> list[PrivateMessage] | None:
        async with self.condition:
            deadline = anyio.current_time() + self.merge_window_seconds
            while True:
                if self.latest_message_id != message_id:
                    return None
                remaining = deadline - anyio.current_time()
                if remaining <= 0:
                    break
                with anyio.move_on_after(remaining):
                    await self.condition.wait()

            while True:
                while True:
                    if self.latest_message_id != message_id:
                        return None
                    if not self.running and not self.pending_image_ids:
                        break
                    await self.condition.wait()

                pending_messages = self.pending_messages.copy()
                has_text = any(
                    message.has_model_visible_content()
                    and message.message_id not in self.dropped_message_ids
                    for message in pending_messages
                )
                messages = [
                    message
                    for message in pending_messages
                    if message.message_id not in self.dropped_message_ids
                    and message.message
                ]
                if not has_text:
                    if not messages:
                        self.pending_messages.clear()
                        self.dropped_message_ids.clear()
                        self.condition.notify_all()
                        return []
                    await self.condition.wait()
                    continue
                self.pending_messages.clear()
                self.dropped_message_ids.clear()
                self.condition.notify_all()
                if not messages:
                    return []

                self.running = True
                return messages

    @property
    def latest_message_id(self) -> str | None:
        if not self.pending_messages:
            return None
        return self.pending_messages[-1].message_id

    async def finish_run(self) -> None:
        async with self.condition:
            self.running = False
            self.condition.notify_all()


class PrivateChatState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    chat: Runnable[dict[str, Any], str] = Field(default_factory=get_chat_app)
    image_analyzer: Runnable[dict[str, Any], str] = Field(
        default_factory=lambda _: get_image_analyzer(True, add_memes_hook=add_memes)
    )
    activity_store: ActivityStore = Field(default_factory=get_activity_store)
    backup_histories: dict[str, deque[str]] = Field(default_factory=dict)
    sessions: dict[str, SessionQueueState] = Field(default_factory=dict)
    sessions_lock: anyio.Lock = Field(default_factory=anyio.Lock)
    image_task_group_lock: anyio.Lock = Field(default_factory=anyio.Lock)
    image_task_group: TaskGroup | None = None
    image_limiter: anyio.CapacityLimiter | None = None


class PrivateChat(Node[PrivateMessageEvent, PrivateChatState, PrivateChatConfig]):
    priority = 1

    @override
    def __init_state__(self) -> PrivateChatState:
        return PrivateChatState()

    def history(self, session_id: str) -> deque[str]:
        return self.node_state.backup_histories.setdefault(
            session_id,
            deque(maxlen=BACKUP_MESSAGES_LIMIT),
        )

    async def session_state(self, session_id: str) -> SessionQueueState:
        async with self.node_state.sessions_lock:
            return self.node_state.sessions.setdefault(
                session_id,
                SessionQueueState(
                    merge_window_seconds=self.config.merge_window_seconds
                ),
            )

    def image_limiter(self) -> anyio.CapacityLimiter:
        limiter = self.node_state.image_limiter
        if (
            limiter is None
            or limiter.total_tokens != self.config.image_analyzer_workers
        ):
            limiter = self.node_state.image_limiter = anyio.CapacityLimiter(
                self.config.image_analyzer_workers
            )
        return limiter

    async def ensure_image_task_group(self) -> TaskGroup:
        async with self.node_state.image_task_group_lock:
            if self.node_state.image_task_group is None:
                self.node_state.image_task_group = anyio.create_task_group()
                await self.node_state.image_task_group.__aenter__()
            return self.node_state.image_task_group

    async def analyze_image_message(self, message: PrivateMessage) -> str | None:
        image, as_meme = message.images[0]
        try:
            async with self.image_limiter():
                return await self.node_state.image_analyzer.ainvoke(
                    {
                        "image": image.base64,
                        "phash": image.phash,
                        "as_meme": as_meme,
                        "detail": False,
                    }
                )
        except Exception:
            return None

    async def start_image_analysis(
        self,
        session_state: SessionQueueState,
        message: PrivateMessage,
    ) -> None:
        async def _finish_image_analysis() -> None:
            await session_state.finish_image(
                message,
                await self.analyze_image_message(message),
            )

        try:
            task_group = await self.ensure_image_task_group()
            task_group.start_soon(_finish_image_analysis)
        except Exception:
            await session_state.finish_image(message, None)

    def activity_weight(self, event_time: int) -> float:
        hour = (
            datetime.fromtimestamp(event_time, tz=UTC)
            .astimezone(ZoneInfo("Asia/Shanghai"))
            .hour
        )
        if self.config.activity_day[0] <= hour < self.config.activity_day[1]:
            return self.config.activity_day_multiplier
        return self.config.activity_night_multiplier

    async def delete_chat(self, session_id: str) -> None:
        await clear_session_history(session_id)
        await self.node_state.activity_store.clear_session(
            scope="private",
            session_id=session_id,
        )
        async with self.node_state.sessions_lock:
            self.node_state.sessions.pop(session_id, None)
        self.node_state.backup_histories.pop(session_id, None)
        await self.reply("[SYSTEM]已清除对话历史。")

    async def record_user_context(
        self,
        session_id: str,
        messages: list[PrivateMessage],
    ) -> None:
        texts = [message.message.get_msgcode() for message in messages]
        self.history(session_id).append("\n".join(texts))
        await get_session_history(session_id).aadd_messages(
            [message.as_human_message() for message in messages]
        )

    async def run_chat(
        self,
        private_event: PrivateEvent,
        messages: list[PrivateMessage],
    ) -> bool:
        history = self.history(private_event.session_id)
        texts = [message.message.get_msgcode() for message in messages]
        history.append("\n".join(texts))
        replied = False
        answer = ""
        print(f"Invoking-Private: {texts}")
        async for reply in self.node_state.chat.astream(
            {
                "messages": messages,
                "extra_prompt": await get_extra_prompt(
                    "\n".join(list(history)[-EXTRA_PROMPT_MAX_HISTORY:])
                ),
                "now_time": datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "user_name": private_event.message.user,
                "thinking": True,
                "reasoning_effort": "max",
            },
            config={"configurable": {"session_id": private_event.session_id}},
        ):
            if reply is None:
                continue
            if not (reply_msg := reply.strip()):
                continue
            print(f"Reply-Private: {reply_msg}")
            raw_cq_msg = await QQMessage.from_str(reply_msg).get_cqhttp_message(
                messages
            )
            for seg in raw_cq_msg:
                if seg.type == "image":
                    await self.reply(seg)
                    break
            else:
                if raw_cq_msg:
                    await self.reply(raw_cq_msg)
            answer += reply_msg + "\n"
            replied = True
        if replied:
            history.append(answer)
        return replied

    async def get_image(self, file: str) -> ImageReadResult | None:
        try:
            result: dict[str, str] = await self.event.adapter.call_api(
                "get_image",
                file=file,
            )
            if path := result.get("file"):
                return await read_image(path, result.get("url"))
        except Exception:
            return None
        return None

    async def get_event(self) -> PrivateEvent | None:
        session_id = self.event.get_session_id()
        text = self.event.get_plain_text()

        if text and any(keyword in text for keyword in self.config.clear_keywords):
            await self.delete_chat(session_id)
            return None

        message = await QQMessage.from_cqhttp_message(
            self.event.message,
            [],
            self.get_image,
        )

        if self.event.reply and (reply_time := int(self.event.reply.time)):
            time_text = (
                datetime.fromtimestamp(reply_time, tz=UTC)
                .astimezone(ZoneInfo("Asia/Shanghai"))
                .strftime("%Y-%m-%d %H:%M:%S")
            )
            message = QQMessageSegment.reply(time_text) + message
        if not message:
            return None

        msg_data = {
            "timestamp": datetime.fromtimestamp(self.event.time, tz=UTC),
            "user": self.event.sender.nickname or "",
            "user_id": str(self.event.user_id),
            "message_id": str(self.event.message_id),
            "message": message,
        }
        images: list[tuple[ImageReadResult, bool]] = []
        msg_with_image = QQMessage()
        for seg in message:
            if seg.type in IMAGE_SEGMENT_TYPES and not seg.data.get("content"):
                image_data = seg.data.get("image")
                if image_data is not None:
                    images.append(
                        (cast("ImageReadResult", image_data), seg.type == "meme")
                    )
                    msg_with_image += seg
            else:
                msg_with_image += seg
        msg = (
            PrivateMessage(**(msg_data | {"message": msg_with_image, "images": images}))
            if images
            else PrivateMessage(**msg_data)
        )
        if not msg.message:
            return None

        return PrivateEvent(
            session_id=session_id,
            message=msg,
        )

    async def claim_messages(
        self,
        private_event: PrivateEvent,
    ) -> tuple[SessionQueueState, list[PrivateMessage]] | None:
        state = await self.session_state(private_event.session_id)
        image_message = await state.enqueue(
            private_event.message,
            self.config.image_hash_similarity_threshold,
        )
        if private_event.message.images and image_message is None:
            return None
        if image_message is not None:
            await self.start_image_analysis(state, image_message)
        messages = await state.wait_and_claim(private_event.message.message_id)
        if not messages:
            return None
        return state, messages

    async def run_reply(
        self,
        private_event: PrivateEvent,
        messages: list[PrivateMessage],
    ) -> None:
        if await self.node_state.activity_store.is_limited(
            scope="private",
            session_id=private_event.session_id,
            event_time=private_event.time,
            activity_limits=self.config.activity_limits,
        ):
            await self.record_user_context(private_event.session_id, messages)
            return

        if await self.run_chat(private_event, messages):
            await self.node_state.activity_store.record(
                scope="private",
                session_id=private_event.session_id,
                event_time=private_event.time,
                weight=self.activity_weight(private_event.time),
                activity_limits=self.config.activity_limits,
            )

    @override
    async def handle(self) -> None:
        private_event = await self.get_event()
        if private_event is None:
            return
        result = await self.claim_messages(private_event)
        if result is None:
            return
        state, messages = result
        try:
            await self.run_reply(private_event, messages)
        finally:
            await state.finish_run()
