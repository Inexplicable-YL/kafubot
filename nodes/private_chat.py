from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from typing_extensions import override
from zoneinfo import ZoneInfo

import anyio
import imagehash
from anyio.abc import TaskGroup
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from pydantic import model_validator
from sekaibot import Node
from sekaibot.adapter.cqhttp.event import PrivateMessageEvent
from sekaibot.adapter.cqhttp.message import CQHTTPMessage, CQHTTPMessageSegment
from sekaibot.config import ConfigModel

from chat.activity import get_activity_store
from chat.image import (
    ImageReadResult,
    get_image_analyzer,
    read_image,
    search_meme,
)
from chat.private import (
    clear_session_history,
    get_chat_app,
    get_session_history,
)
from chat.prompt import get_extra_prompt
from chat.utils import parse_message

BACKUP_MESSAGES_LIMIT = 10

EXTRA_PROMPT_MAX_HISTORY = 3


class PrivateChatConfig(ConfigModel):
    """私聊记录节点配置"""

    __config_name__ = "private_chat"
    merge_window_seconds: float = 5
    image_analyzer_workers: int = 10
    image_hash_similarity_threshold: int = 5
    activity_day: tuple[int, int] = (8, 20)  # 8:00-20:00 is considered day time
    # (window_seconds, threshold)
    activity_limits: tuple[tuple[int, int], ...] = (
        (3600 * 5, 100),
        (3600 * 24 * 7, 450),
    )
    activity_day_multiplier: float = 0.5
    activity_night_multiplier: float = 1.0

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


@dataclass
class QueuedMessage:
    content: str | None = None
    has_text: bool = False
    timestamp_text: str = ""
    image_hash: imagehash.ImageHash | None = None
    pending_image: bool = False
    dropped: bool = False


@dataclass
class SessionQueueState:
    running: bool = False
    pending_messages: list[QueuedMessage] = field(default_factory=list)
    latest_waiter_id: int = 0
    last_enqueue_time: float | None = None
    merge_window_seconds: float = 5
    pending_image_count: int = 0
    condition: anyio.Condition = field(default_factory=anyio.Condition)

    async def enqueue_text(self, message: str) -> int:
        async with self.condition:
            self.pending_messages.append(QueuedMessage(content=message, has_text=True))
            self.latest_waiter_id += 1
            self.last_enqueue_time = anyio.current_time()
            waiter_id = self.latest_waiter_id
            self.condition.notify_all()
            return waiter_id

    async def enqueue_image_if_unique(
        self,
        timestamp_text: str,
        image_hash: imagehash.ImageHash,
        similarity_threshold: int,
    ) -> tuple[int, QueuedMessage] | None:
        async with self.condition:
            if self.has_similar_image_locked(image_hash, similarity_threshold):
                return None
            message = QueuedMessage(
                timestamp_text=timestamp_text,
                image_hash=image_hash,
                pending_image=True,
            )
            self.pending_messages.append(message)
            self.pending_image_count += 1
            self.latest_waiter_id += 1
            self.last_enqueue_time = anyio.current_time()
            waiter_id = self.latest_waiter_id
            self.condition.notify_all()
            return waiter_id, message

    def has_similar_image_locked(
        self,
        image_hash: imagehash.ImageHash,
        similarity_threshold: int,
    ) -> bool:
        return any(
            message.image_hash is not None
            and not message.dropped
            and message.image_hash - image_hash <= similarity_threshold
            for message in self.pending_messages
        )

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
                message.content = f"{message.timestamp_text}[图片: {abstract}]"
            else:
                message.dropped = True
                message.content = None
            self.condition.notify_all()

    async def wait_and_claim(self, waiter_id: int) -> list[str] | None:
        async with self.condition:
            while True:
                if waiter_id != self.latest_waiter_id:
                    return None

                if (
                    not self.running
                    and self.pending_messages
                    and self.last_enqueue_time is not None
                ):
                    elapsed = anyio.current_time() - self.last_enqueue_time
                    remaining = self.merge_window_seconds - elapsed
                    if remaining > 0:
                        # Wait for remaining time or until notified (whichever comes first)
                        with anyio.move_on_after(remaining):
                            await self.condition.wait()
                        # Recalculate on next iteration if timeout fired
                        continue
                    if self.pending_image_count > 0:
                        await self.condition.wait()
                        continue

                    pending_messages = self.pending_messages.copy()
                    has_text = any(
                        message.has_text and not message.dropped
                        for message in pending_messages
                    )
                    merged_messages = [
                        message.content
                        for message in pending_messages
                        if not message.dropped and message.content is not None
                    ]
                    if not has_text:
                        if not merged_messages:
                            self.pending_messages.clear()
                            self.condition.notify_all()
                            return []
                        await self.condition.wait()
                        continue

                    self.pending_messages.clear()
                    self.running = True
                    return merged_messages

                await self.condition.wait()

    async def finish_run(self) -> None:
        async with self.condition:
            self.running = False
            self.condition.notify_all()


@dataclass
class ImageAnalyzeJob:
    session_state: SessionQueueState
    message: QueuedMessage
    image: str
    phash: imagehash.ImageHash
    as_meme: bool = False


@dataclass
class PrivateReplyState:
    chat_app: Runnable[dict[str, Any], str] = field(default_factory=get_chat_app)
    image_analyzer: Runnable[dict[str, Any], str] = field(
        default_factory=get_image_analyzer
    )
    backup_histories: dict[str, list[str]] = field(default_factory=dict)
    sessions: dict[str, SessionQueueState] = field(default_factory=dict)
    sessions_lock: anyio.Lock = field(default_factory=anyio.Lock)
    image_worker_lock: anyio.Lock = field(default_factory=anyio.Lock)
    image_worker_task_group: TaskGroup | None = None
    image_worker_count: int = 0
    image_job_send_stream: MemoryObjectSendStream[ImageAnalyzeJob] = field(init=False)
    image_job_receive_stream: MemoryObjectReceiveStream[ImageAnalyzeJob] = field(
        init=False
    )

    def __post_init__(self) -> None:
        send_stream, receive_stream = anyio.create_memory_object_stream[
            ImageAnalyzeJob
        ](100)
        self.image_job_send_stream = send_stream
        self.image_job_receive_stream = receive_stream


class PrivateChat(Node[PrivateMessageEvent, PrivateReplyState, PrivateChatConfig]):  # type: ignore
    priority = 1

    def __init_state__(self) -> PrivateReplyState:
        return PrivateReplyState()

    def _event_timestamp_text(self) -> str:
        return f"[{datetime.fromtimestamp(self.event.time, tz=UTC).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')}]"

    async def _get_session_state(self, session_id: str) -> SessionQueueState:
        async with self.node_state.sessions_lock:
            if session_id not in self.node_state.sessions:
                self.node_state.sessions[session_id] = SessionQueueState(
                    merge_window_seconds=self.config.merge_window_seconds
                )
            return self.node_state.sessions[session_id]

    @staticmethod
    async def _image_worker(node_state: PrivateReplyState) -> None:
        async with node_state.image_job_receive_stream.clone() as receive_stream:
            async for job in receive_stream:
                abstract: str | None = None
                try:
                    result = await node_state.image_analyzer.ainvoke(
                        {
                            "image": job.image,
                            "phash": job.phash,
                            "as_meme": job.as_meme,
                            "detail": False,
                        }
                    )
                    abstract = result or None
                except Exception:
                    abstract = None
                await job.session_state.finish_image(job.message, abstract)

    async def _ensure_image_workers(self) -> None:
        async with self.node_state.image_worker_lock:
            if self.node_state.image_worker_task_group is None:
                self.node_state.image_worker_task_group = anyio.create_task_group()
                await self.node_state.image_worker_task_group.__aenter__()
                self.node_state.image_worker_count = 0

            for _ in range(
                self.config.image_analyzer_workers - self.node_state.image_worker_count
            ):
                self.node_state.image_worker_task_group.start_soon(
                    self._image_worker,
                    self.node_state,
                )
                self.node_state.image_worker_count += 1

    def _activity_weight(self, event_time: int) -> float:
        background_time = datetime.fromtimestamp(
            event_time,
            tz=UTC,
        ).astimezone(ZoneInfo("Asia/Shanghai"))
        if (
            self.config.activity_day[0]
            <= background_time.hour
            < self.config.activity_day[1]
        ):
            return self.config.activity_day_multiplier
        return self.config.activity_night_multiplier

    async def _is_activity_limited(self, session_id: str, event_time: int) -> bool:
        return await get_activity_store().is_limited(
            scope="private",
            session_id=session_id,
            event_time=event_time,
            activity_limits=self.config.activity_limits,
        )

    async def _delete_chat(self, session_id: str) -> None:
        await clear_session_history(session_id)
        await get_activity_store().clear_session(scope="private", session_id=session_id)
        async with self.node_state.sessions_lock:
            if session_id in self.node_state.sessions:
                del self.node_state.sessions[session_id]
        await self.reply("[SYSTEM]已清除对话历史。")

    async def _record_user_context(self, messages: list[str], session_id: str) -> None:
        self.node_state.backup_histories.setdefault(session_id, [])
        joined_messages = "\n".join(messages)
        self.node_state.backup_histories[session_id].append(joined_messages)
        self.node_state.backup_histories[session_id] = self.node_state.backup_histories[
            session_id
        ][-BACKUP_MESSAGES_LIMIT:]
        await get_session_history(session_id).aadd_messages(
            [HumanMessage(content=message) for message in messages]
        )

    async def _run_chat(self, messages: list[str], name: str, session_id: str) -> bool:
        self.node_state.backup_histories.setdefault(session_id, [])
        self.node_state.backup_histories[session_id].append("\n".join(messages))
        replied = False
        answer = ""
        print(f"Invoking: {messages}")
        async for reply in self.node_state.chat_app.astream(
            {
                "messages": messages,
                "extra_prompt": await get_extra_prompt(
                    "\n".join(
                        self.node_state.backup_histories[session_id][
                            -EXTRA_PROMPT_MAX_HISTORY:
                        ]
                    )
                ),
                "now_time": datetime.now(tz=ZoneInfo("Asia/Shanghai")).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "user_name": name,
                "thinking": True,
                "reasoning_effort": "max",
            },
            config={"configurable": {"session_id": session_id}},
        ):
            if reply is not None:
                print(f"Reply-Private: {reply}")
                segments, text = parse_message(reply.strip())
                message: CQHTTPMessage | str = ""
                for seg in segments:
                    if seg.type == "meme" and "content" in seg.data:
                        content = seg.data["content"]
                        meme_result = await search_meme(
                            content, temperature=0.15, min_score=0.05
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
                answer += reply
                replied = True
        if replied:
            self.node_state.backup_histories[session_id].append(answer)
        self.node_state.backup_histories[session_id] = self.node_state.backup_histories[
            session_id
        ][-BACKUP_MESSAGES_LIMIT:]
        return replied

    async def get_image(self, file: str) -> tuple[ImageReadResult | None, bool]:
        try:
            result: dict[str, str] = await self.event.adapter.call_api(
                "get_image", file=file
            )
            path, url, sub_type = (
                result.get("file"),
                result.get("url"),
                result.get("sub_type"),
            )
            if path is not None:
                return await read_image(path, url), str(sub_type) == "1"
        except Exception:
            return None, False
        return None, False

    @override
    async def handle(self) -> None:
        raw_text = self.event.get_plain_text()
        timestamp_text = self._event_timestamp_text()
        session_id = self.event.get_session_id()
        text: str | None = None
        image: ImageReadResult | None = None
        as_meme: bool = False

        if raw_text:
            text = timestamp_text + raw_text

            keyws = [
                "clear",
                "清除",
                "清空",
                "清理",
                "删除",
                "重置",
                "重新开始",
                "重启",
            ]
            if any(keyw in text for keyw in keyws):
                await self._delete_chat(session_id=session_id)
                return
        else:
            file: str | None = None
            if len(self.event.message) == 1 and self.event.message[0].type == "image":
                file = self.event.message[0].data.get("file")
            if file is not None:
                image, as_meme = await self.get_image(file)
            if image is None:
                return

        name_map: dict[str, str] = {
            "shiroko": "白子小姐",
            "空想少女": "アルス",
            "かたちなきもの": "言霊",
        }
        name: str = self.event.sender.nickname or ""
        name = name_map.get(name, name)

        session_state = await self._get_session_state(session_id)

        if text is not None:
            decision = await session_state.enqueue_text(text)
        elif image is not None:
            await self._ensure_image_workers()
            image_enqueue_result = await session_state.enqueue_image_if_unique(
                timestamp_text=timestamp_text,
                image_hash=image.phash,
                similarity_threshold=self.config.image_hash_similarity_threshold,
            )
            if image_enqueue_result is None:
                return
            decision, image_message = image_enqueue_result
            try:
                await self.node_state.image_job_send_stream.send(
                    ImageAnalyzeJob(
                        session_state=session_state,
                        message=image_message,
                        image=image.base64,
                        phash=image.phash,
                        as_meme=as_meme,
                    )
                )
            except Exception:
                await session_state.finish_image(image_message, None)
        else:
            return

        messages = await session_state.wait_and_claim(decision)
        if not messages:
            return

        try:
            if await self._is_activity_limited(session_id, self.event.time):
                await self._record_user_context(
                    messages=messages,
                    session_id=session_id,
                )
                return

            replied = await self._run_chat(
                messages=messages,
                name=name,
                session_id=session_id,
            )
            if replied:
                await get_activity_store().record(
                    scope="private",
                    session_id=session_id,
                    event_time=self.event.time,
                    weight=self._activity_weight(self.event.time),
                    activity_limits=self.config.activity_limits,
                )
        finally:
            await session_state.finish_run()
