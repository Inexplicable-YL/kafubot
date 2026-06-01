from datetime import UTC, datetime
from typing import Any, cast
from typing_extensions import override
from zoneinfo import ZoneInfo

import anyio
from anyio.abc import TaskGroup
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sekaibot import Bot, Node
from sekaibot.adapter.cqhttp.event import PrivateMessageEvent
from sekaibot.config import ConfigModel

from agent import (
    UserMessage,
    clear_session_history,
    create_agent_service,
)
from agent.base import ManagerContext, ManagerState
from agent.message import QQMessage, QQMessageSegment
from agent.multimodal.image import (
    ImageReadResult,
    get_analyzer,
    read_image,
)
from agent.multimodal.meme import add_memes

LIMITER_DB = "sqlite+aiosqlite:///./.database/private_limiter.db"
DEFAULT_CLEAR_KEYWORDS = {"/clear", "/清除"}


def _has_model_visible_content(message: UserMessage) -> bool:
    return any(
        bool(seg.data.get("text", "").strip())
        if seg.type == "text"
        else seg.type not in {"image", "meme"} or bool(seg.data.get("content"))
        for seg in message.message
    )


class PrivateAgentConfig(ConfigModel):
    """私聊记录节点配置"""

    __config_name__ = "private_agent"

    merge_window_seconds: float = 5
    image_analyzer_workers: int = 10
    image_hash_similarity_threshold: int = 5
    talk_value: float = 0.8
    reply_keywords: set[tuple[str, float]] = set()
    clear_keywords: set[str] = set()
    average_reply_count: float = 1.5
    # (requests_per_second, max_bucket_size)
    rate_limit: tuple[float, float] = (1.0, 1.0)
    # (window_seconds, threshold)
    activity_limits: tuple[tuple[int, int], ...] = (
        (3600 * 5, 100),
        (3600 * 24 * 7, 500),
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
        self.clear_keywords = self.clear_keywords.union(DEFAULT_CLEAR_KEYWORDS)
        return self


class SessionQueueState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    running: bool = False
    pending_messages: list[UserMessage] = Field(default_factory=list)
    merge_window_seconds: float = 5
    pending_image_ids: set[str] = Field(default_factory=set)
    dropped_message_ids: set[str] = Field(default_factory=set)
    condition: anyio.Condition = Field(default_factory=anyio.Condition)

    async def enqueue(
        self,
        message: UserMessage,
        similarity_threshold: int,
    ) -> UserMessage | None:
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
        message: UserMessage,
        abstracts: list[str | None],
    ) -> None:
        async with self.condition:
            self.pending_image_ids.discard(message.message_id)
            if any(abstracts) or _has_model_visible_content(message):
                new_message = QQMessage()
                content_iter = iter(abstracts)
                for seg in message.message:
                    if seg.type in {"image", "meme"} and not seg.data.get("content"):
                        content = next(content_iter, None)
                        if content:
                            data = dict(seg.data)
                            data["content"] = content
                            new_message += getattr(QQMessageSegment, seg.type)(**data)
                        continue
                    new_message += seg
                message.message = new_message
                if not message.message:
                    self.dropped_message_ids.add(message.message_id)
            else:
                self.dropped_message_ids.add(message.message_id)
            self.condition.notify_all()

    async def wait_and_claim(self, message_id: str) -> list[UserMessage] | None:
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
                    _has_model_visible_content(message)
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


class PrivateAgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Any | None = None
    image_analyzer: Runnable[dict[str, Any], str] = Field(
        default_factory=lambda _: get_analyzer(True, add_memes_hook=add_memes)
    )
    sessions: dict[str, SessionQueueState] = Field(default_factory=dict)
    sessions_lock: anyio.Lock = Field(default_factory=anyio.Lock)
    image_task_group_lock: anyio.Lock = Field(default_factory=anyio.Lock)
    image_task_group: TaskGroup | None = None
    image_limiter: anyio.CapacityLimiter | None = None


class PrivateAgent(Node[PrivateMessageEvent, PrivateAgentState, PrivateAgentConfig]):
    priority = 1

    @override
    def __init_state__(self) -> PrivateAgentState:
        return PrivateAgentState()

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

    async def analyze_image_message(self, message: UserMessage) -> list[str | None]:
        results: list[str | None] = [None] * len(message.images)

        async def _analyze_image(
            index: int, image: ImageReadResult, as_meme: bool
        ) -> None:
            try:
                async with self.image_limiter():
                    results[index] = await self.node_state.image_analyzer.ainvoke(
                        {
                            "image": image.base64,
                            "phash": image.phash,
                            "as_meme": as_meme,
                        }
                    )
            except Exception:
                results[index] = None

        async with anyio.create_task_group() as tg:
            for index, (image, as_meme) in enumerate(message.images):
                tg.start_soon(_analyze_image, index, image, as_meme)
        return results

    async def start_image_analysis(
        self,
        session_state: SessionQueueState,
        message: UserMessage,
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
            await session_state.finish_image(message, [None] * len(message.images))

    async def delete_chat(self, session_id: str) -> None:
        await clear_session_history(session_id)
        async with self.node_state.sessions_lock:
            self.node_state.sessions.pop(session_id, None)
        await self.reply("[SYSTEM]已清除对话历史。")

    async def run_chat(
        self,
        session_id: str,
        messages: list[UserMessage],
    ) -> None:
        print(
            f"Private-Agent-Invoking: 已省略{len(messages) - 5}个消息，{[m.message.get_msgcode() for m in messages][-5:]}"
        )
        if not self.node_state.agent:
            get_agent, close_agent = await create_agent_service()
            Bot.bot_exit_hook(close_agent)
            self.node_state.agent = await get_agent(
                interaction_config={
                    "average_reply_count": self.config.average_reply_count,
                    "stop_when_reply": True,
                },
                gate_config={
                    "talk_value": self.config.talk_value,
                    "keywords": set(self.config.reply_keywords),
                },
                limiter_config={
                    "db_url": LIMITER_DB,
                    "rate_limit": self.config.rate_limit,
                    "act_limits": self.config.activity_limits,
                },
            )
        assert self.node_state.agent
        await self.node_state.agent.ainvoke(
            ManagerState(
                messages=[],
                inputs=messages,
                outputs=[],
                currents=[],
                histories=[],
            ),
            context=ManagerContext(
                session_id=session_id,
                is_tome=True,
                node=self,
                chat_id=str(self.event.user_id),
                unrestricted=False,
            ),
        )

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

    async def get_message(self, session_id: str) -> UserMessage | None:
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
            "is_tome": True,
        }
        images: list[tuple[ImageReadResult, bool]] = []
        msg_with_image = QQMessage()
        for seg in message:
            if seg.type in {"image", "meme"} and not seg.data.get("content"):
                image_data = seg.data.get("image")
                if image_data is not None:
                    images.append(
                        (cast("ImageReadResult", image_data), seg.type == "meme")
                    )
                    msg_with_image += seg
            else:
                msg_with_image += seg
        msg = (
            UserMessage(**(msg_data | {"message": msg_with_image, "images": images}))
            if images
            else UserMessage(**msg_data)
        )
        if not msg.message:
            return None

        return msg

    async def claim_messages(
        self,
        session_id: str,
        message: UserMessage,
    ) -> tuple[SessionQueueState, list[UserMessage]] | None:
        state = await self.session_state(session_id)
        image_message = await state.enqueue(
            message,
            self.config.image_hash_similarity_threshold,
        )
        if message.images and image_message is None:
            return None
        if image_message is not None:
            await self.start_image_analysis(state, image_message)
        messages = await state.wait_and_claim(message.message_id)
        if not messages:
            return None
        return state, messages

    @override
    async def handle(self) -> None:
        session_id = self.event.get_session_id()
        message = await self.get_message(session_id)
        if message is None:
            return
        result = await self.claim_messages(session_id, message)
        if result is None:
            return
        state, messages = result
        try:
            await self.run_chat(session_id, messages)
        finally:
            await state.finish_run()
