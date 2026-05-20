from collections import deque
from datetime import UTC, datetime
from typing import Any, TypeAlias
from typing_extensions import override
from zoneinfo import ZoneInfo

import anyio
import imagehash
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sekaibot import Node
from sekaibot.adapter.cqhttp.event import PrivateMessageEvent
from sekaibot.adapter.cqhttp.message import CQHTTPMessage, CQHTTPMessageSegment
from sekaibot.config import ConfigModel

from chat.activity import ActivityStore, get_activity_store
from chat.image import (
    ImageReadResult,
    get_image_analyzer,
    read_image,
)
from chat.meme import add_memes, search_meme
from chat.private import (
    clear_session_history,
    get_chat_app,
    get_session_history,
)
from chat.prompt import get_extra_prompt
from chat.utils import parse_message

BACKUP_MESSAGES_LIMIT = 10
EXTRA_PROMPT_MAX_HISTORY = 3
IMAGE_ANALYZE_QUEUE_SIZE = 100


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
    message_id: int
    timestamp: datetime
    user_name: str
    text: str | None = None
    image: ImageReadResult | None = None
    as_meme: bool = False

    @property
    def time(self) -> int:
        return int(self.timestamp.timestamp())


class QueuedMessage(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_id: int
    timestamp: datetime
    text: str | None = None
    has_text: bool = False
    image_hash: imagehash.ImageHash | None = None
    pending_image: bool = False
    dropped: bool = False


class SessionQueueState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    running: bool = False
    pending_messages: list[QueuedMessage] = Field(default_factory=list)
    merge_window_seconds: float = 5
    pending_image_count: int = 0
    condition: anyio.Condition = Field(default_factory=anyio.Condition)

    async def enqueue_text(self, private_event: PrivateEvent) -> None:
        if private_event.text is None:
            return
        async with self.condition:
            self.pending_messages.append(
                QueuedMessage(
                    message_id=private_event.message_id,
                    timestamp=private_event.timestamp,
                    text=private_event.text,
                    has_text=True,
                )
            )
            self.condition.notify_all()

    async def enqueue_image(
        self,
        private_event: PrivateEvent,
        similarity_threshold: int,
    ) -> QueuedMessage | None:
        if private_event.image is None:
            return None
        async with self.condition:
            if any(
                message.image_hash is not None
                and not message.dropped
                and message.image_hash - private_event.image.phash
                <= similarity_threshold
                for message in self.pending_messages
            ):
                return None
            message = QueuedMessage(
                message_id=private_event.message_id,
                timestamp=private_event.timestamp,
                image_hash=private_event.image.phash,
                pending_image=True,
            )
            self.pending_messages.append(message)
            self.pending_image_count += 1
            self.condition.notify_all()
            return message

    async def finish_image(
        self,
        message: QueuedMessage,
        abstract: str | None,
    ) -> None:
        async with self.condition:
            if message.pending_image:
                message.pending_image = False
                self.pending_image_count = max(0, self.pending_image_count - 1)
            if abstract:
                message.text = f"[图片: {abstract}]"
            else:
                message.dropped = True
                message.text = None
            self.condition.notify_all()

    async def wait_and_claim(self, message_id: int) -> list[QueuedMessage] | None:
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
                    if not self.running and self.pending_image_count == 0:
                        break
                    await self.condition.wait()

                pending_messages = self.pending_messages.copy()
                has_text = any(
                    message.has_text and not message.dropped
                    for message in pending_messages
                )
                messages = [
                    message
                    for message in pending_messages
                    if not message.dropped and message.text is not None
                ]
                if not has_text:
                    if not messages:
                        self.pending_messages.clear()
                        self.condition.notify_all()
                        return []
                    await self.condition.wait()
                    continue
                self.pending_messages.clear()
                self.condition.notify_all()
                if not messages:
                    return []

                self.running = True
                return messages

    @property
    def latest_message_id(self) -> int | None:
        if not self.pending_messages:
            return None
        return self.pending_messages[-1].message_id

    async def finish_run(self) -> None:
        async with self.condition:
            self.running = False
            self.condition.notify_all()


ImageAnalyzeJob: TypeAlias = tuple[
    SessionQueueState,
    QueuedMessage,
    str,
    imagehash.ImageHash,
    bool,
]


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
    image_worker_lock: anyio.Lock = Field(default_factory=anyio.Lock)
    image_worker_task_group: TaskGroup | None = None
    image_worker_count: int = 0
    image_job_send_stream: MemoryObjectSendStream[ImageAnalyzeJob]
    image_job_receive_stream: MemoryObjectReceiveStream[ImageAnalyzeJob]


class PrivateChat(Node[PrivateMessageEvent, PrivateChatState, PrivateChatConfig]):
    priority = 1

    @override
    def __init_state__(self) -> PrivateChatState:
        send_stream, receive_stream = anyio.create_memory_object_stream[
            ImageAnalyzeJob
        ](IMAGE_ANALYZE_QUEUE_SIZE)
        return PrivateChatState(
            image_job_send_stream=send_stream,
            image_job_receive_stream=receive_stream,
        )

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

    @staticmethod
    async def image_worker(node_state: PrivateChatState) -> None:
        async with node_state.image_job_receive_stream.clone() as receive_stream:
            async for job in receive_stream:
                session_state, message, image, phash, as_meme = job
                abstract: str | None = None
                try:
                    abstract = await node_state.image_analyzer.ainvoke(
                        {
                            "image": image,
                            "phash": phash,
                            "as_meme": as_meme,
                            "detail": False,
                        }
                    )
                except Exception:
                    abstract = None
                await session_state.finish_image(message, abstract)

    async def ensure_image_workers(self) -> None:
        async with self.node_state.image_worker_lock:
            if self.node_state.image_worker_task_group is None:
                self.node_state.image_worker_task_group = anyio.create_task_group()
                await self.node_state.image_worker_task_group.__aenter__()

            while (
                self.node_state.image_worker_count < self.config.image_analyzer_workers
            ):
                self.node_state.image_worker_task_group.start_soon(
                    self.image_worker,
                    self.node_state,
                )
                self.node_state.image_worker_count += 1

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
        messages: list[QueuedMessage],
    ) -> None:
        texts = [message.text for message in messages if message.text is not None]
        self.history(session_id).append("\n".join(texts))
        await get_session_history(session_id).aadd_messages(
            [
                HumanMessage(
                    content=message.text,
                    additional_kwargs={
                        "raw": {
                            "timestamp": message.timestamp,
                            "text": message.text,
                        }
                    },
                )
                for message in messages
                if message.text is not None
            ]
        )

    async def run_chat(
        self,
        private_event: PrivateEvent,
        messages: list[QueuedMessage],
    ) -> bool:
        history = self.history(private_event.session_id)
        texts = [message.text for message in messages if message.text is not None]
        history.append("\n".join(texts))
        replied = False
        answer = ""
        print(f"Invoking-Private: {texts}")
        async for reply in self.node_state.chat.astream(
            {
                "messages": [
                    message.model_dump(include={"timestamp", "text"})
                    for message in messages
                ],
                "extra_prompt": await get_extra_prompt(
                    "\n".join(list(history)[-EXTRA_PROMPT_MAX_HISTORY:])
                ),
                "now_time": datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "user_name": private_event.user_name,
                "thinking": True,
                "reasoning_effort": "max",
            },
            config={"configurable": {"session_id": private_event.session_id}},
        ):
            if reply is None:
                continue
            print(f"Reply-Private: {reply}")
            segments, text = parse_message(reply.strip())
            message: CQHTTPMessage | str = ""
            for seg in segments:
                if seg.type != "meme" or "content" not in seg.data:
                    continue
                meme_result = await search_meme(
                    seg.data["content"],
                    temperature=0.5,
                    min_score=0.0,
                )
                if meme_result:
                    await self.reply(
                        CQHTTPMessageSegment.image(meme_result.base64, sub_type=1)
                    )
            message += text
            if message:
                await self.reply(message)
            answer += reply
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
        image: ImageReadResult | None = None
        as_meme = False

        if text:
            if any(keyword in text for keyword in self.config.clear_keywords):
                await self.delete_chat(session_id)
                return None
        elif len(self.event.message) == 1 and self.event.message[0].type == "image":
            image_segment = self.event.message[0]
            file = image_segment.data.get("file")
            image = await self.get_image(file) if file else None
            as_meme = str(image_segment.data.get("sub_type", "0")) == "1"
            if image is None:
                return None
        else:
            return None

        return PrivateEvent(
            session_id=session_id,
            message_id=self.event.message_id,
            timestamp=datetime.fromtimestamp(self.event.time, tz=UTC),
            user_name=self.event.sender.nickname or "",
            text=text or None,
            image=image,
            as_meme=as_meme,
        )

    async def claim_messages(
        self,
        private_event: PrivateEvent,
    ) -> tuple[SessionQueueState, list[QueuedMessage]] | None:
        state = await self.session_state(private_event.session_id)
        if private_event.text is not None:
            await state.enqueue_text(private_event)
            messages = await state.wait_and_claim(private_event.message_id)
            if not messages:
                return None
            return state, messages
        await self.ensure_image_workers()
        image_result = await state.enqueue_image(
            private_event,
            self.config.image_hash_similarity_threshold,
        )
        if image_result is None or private_event.image is None:
            return None
        queued_message = image_result
        try:
            await self.node_state.image_job_send_stream.send(
                (
                    state,
                    queued_message,
                    private_event.image.base64,
                    private_event.image.phash,
                    private_event.as_meme,
                )
            )
        except Exception:
            await state.finish_image(queued_message, None)
        messages = await state.wait_and_claim(private_event.message_id)
        if not messages:
            return None
        return state, messages

    async def run_reply(
        self,
        private_event: PrivateEvent,
        messages: list[QueuedMessage],
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
