from collections import deque
from datetime import UTC, datetime
from typing import Any, cast
from typing_extensions import override
from zoneinfo import ZoneInfo

import anyio
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sekaibot import Bot, Node
from sekaibot.adapter.cqhttp.event import GroupMessageEvent
from sekaibot.config import ConfigModel
from sekaibot.log import logger

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

MESSAGES_LIMIT = 30
BACKUP_MESSAGES_LIMIT = 20


LIMITER_DB = "sqlite+aiosqlite:///./.database/group_limiter.db"

DEFAULT_USERNAME = "陌生用户"

DEFAULT_CLEAR_KEYWORDS = {"/clear", "/清除"}


def _extract_reply(agent_output: Any) -> str | None:
    candidates = agent_output.get("outputs")
    full_text = ""
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict) and isinstance(item.get("data"), dict):
                if item.get("type") == "reply":
                    full_text += item["data"].get("full_text") or ""
                elif item.get("type") == "meme":
                    full_text += (
                        f"[MSG:meme, content={item['data'].get('content') or ''}]"
                    )
    return full_text or None


class GroupAgentConfig(ConfigModel):
    __config_name__ = "group_agent"

    unrestricted_groups: set[int] = set()
    auto_reply_groups: set[int] = set()
    keep_image_limit: int = 3
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
    def _add_default(self):
        self.auto_reply_groups = self.auto_reply_groups.union(self.unrestricted_groups)
        self.clear_keywords = self.clear_keywords.union(DEFAULT_CLEAR_KEYWORDS)
        return self


class Histories(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: deque[UserMessage] = Field(
        default_factory=lambda: deque(maxlen=MESSAGES_LIMIT)
    )
    backup_messages: deque[UserMessage] = Field(
        default_factory=lambda: deque(maxlen=BACKUP_MESSAGES_LIMIT)
    )
    on_handle: bool = False
    handle_condition: anyio.Condition = Field(default_factory=anyio.Condition)
    lock: anyio.Lock = Field(default_factory=anyio.Lock)


class GroupEvent(BaseModel):
    session_id: str
    history_storage: Histories
    message: UserMessage

    @property
    def time(self) -> int:
        return int(self.message.timestamp.timestamp())


class GroupAgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Any | None = None
    image_analyzer: Runnable[dict[str, Any], str] = Field(
        default_factory=lambda _: get_analyzer(True, add_memes_hook=add_memes)
    )
    storages: dict[str, Histories] = Field(default_factory=dict)
    storages_lock: anyio.Lock = Field(default_factory=anyio.Lock)


class GroupAgent(Node[GroupMessageEvent, GroupAgentState, GroupAgentConfig]):
    """群聊记录节点"""

    priority = 0

    @override
    def __init_state__(self) -> GroupAgentState:
        return GroupAgentState()

    async def get_history_storage(
        self,
        session_id: str,
    ) -> Histories:
        async with self.node_state.storages_lock:
            if session_id not in self.node_state.storages:
                self.node_state.storages[session_id] = Histories()
            return self.node_state.storages[session_id]

    async def claim_messages(  # noqa: PLR0911
        self,
        group_event: GroupEvent,
    ) -> list[UserMessage] | None:
        history_storage = group_event.history_storage
        message = group_event.message
        current_messages: list[UserMessage] | None = None
        is_tome = message.is_tome or any(
            msg.is_tome for msg in history_storage.messages
        )
        await history_storage.lock.acquire()
        is_released = False
        try:
            history_storage.messages.append(message)
            history_storage.backup_messages.append(message)
            if group_event.message.images:
                return None
            if not group_event.message.message.get_plain_text().strip():
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

            history_storage.on_handle = True
            current_messages = list(history_storage.messages)
            history_storage.messages.clear()
        finally:
            if not is_released:
                history_storage.lock.release()
        return current_messages

    async def filling_images(
        self,
        current_messages: list[UserMessage],
    ) -> list[UserMessage]:
        if not current_messages:
            return []
        skip = (
            sum(bool(x.images) for x in current_messages) - self.config.keep_image_limit
        )
        messages = [
            x for x in current_messages if not x.images or (skip := skip - 1) < 0
        ]
        image_inputs = []
        for x in [x for x in messages if x.images]:
            image_inputs.extend(
                {
                    "image": img[0].base64,
                    "phash": img[0].phash,
                    "as_meme": img[1],
                }
                for img in x.images
            )
        if not image_inputs:
            return messages
        results = await self.node_state.image_analyzer.abatch(image_inputs)
        res_idx = 0
        for img_msg in messages:
            if img_msg.images:
                new_message = QQMessage()
                for seg in img_msg.message:
                    if seg.type in {"image", "meme"} and not seg.data.get("content"):
                        new_message += getattr(QQMessageSegment, seg.type)(
                            content=results[res_idx]
                        )
                        res_idx += 1
                    else:
                        new_message += seg
                img_msg.message = new_message

        return messages

    async def run_reply(
        self,
        group_event: GroupEvent,
        *,
        current_messages: list[UserMessage],
    ) -> list[UserMessage] | None:
        current_messages = await self.filling_images(current_messages)
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
        agent_output = await self.node_state.agent.ainvoke(
            ManagerState(
                messages=[],
                inputs=current_messages,
                outputs=[],
                currents=[],
                histories=[],
            ),
            context=ManagerContext(
                session_id=group_event.session_id,
                is_tome=group_event.message.is_tome,
                node=self,
                chat_id=str(self.event.group_id),
                unrestricted=(self.event.group_id in self.config.unrestricted_groups),
            ),
        )

        if _extract_reply(agent_output):
            return None
        return current_messages

    async def finish_reply(
        self,
        group_event: GroupEvent,
        *,
        current_messages: list[UserMessage] | None = None,
    ) -> None:
        history_storage = group_event.history_storage
        async with history_storage.lock:
            if current_messages:
                new_messages = list(history_storage.messages)
                history_storage.messages.clear()
                history_storage.messages.extend(current_messages + new_messages)
            history_storage.on_handle = False
            async with history_storage.handle_condition:
                history_storage.handle_condition.notify_all()

    async def delete_chat(self, session_id: str) -> None:
        await clear_session_history(session_id=session_id)
        async with self.node_state.storages_lock:
            if session_id in self.node_state.storages:
                del self.node_state.storages[session_id]
        await self.reply("[SYSTEM]已清除历史", at_sender=True)

    async def get_message_or_clear_chat(
        self, session_id: str, history_storage: Histories, to_me: bool
    ) -> QQMessage | None:
        if (text := self.event.message.get_plain_text()) and any(
            keyw in text for keyw in self.config.clear_keywords
        ):
            await self.delete_chat(session_id=session_id)
            return None
        message = await QQMessage.from_cqhttp_message(
            self.event.message, history_storage.backup_messages, self.get_image
        )

        if to_me:
            message = QQMessageSegment.at("可不") + message
        if self.event.reply and (reply_time := int(self.event.reply.time)):
            time_text = (
                datetime.fromtimestamp(reply_time, tz=UTC)
                .astimezone(ZoneInfo("Asia/Shanghai"))
                .strftime("%Y-%m-%d %H:%M:%S")
            )
            message = (
                QQMessageSegment.reply(
                    time_text,
                    str(self.event.reply.message_id),
                    include={"message_id"},
                )
                + message
            )
        return message

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
        session_id = f"group_{self.event.group_id}"
        history_storage = await self.get_history_storage(session_id)
        is_tome = self.event.is_tome()
        message = await self.get_message_or_clear_chat(
            session_id, history_storage, is_tome
        )
        if not message:
            return None
        timestamp = datetime.fromtimestamp(self.event.time, tz=UTC)
        user = self.event.sender.nickname or DEFAULT_USERNAME
        msg_data = {
            "role": "user",
            "timestamp": timestamp,
            "user": user,
            "message": message,
            "user_id": str(self.event.user_id),
            "message_id": str(self.event.message_id),
            "is_tome": is_tome,
        }
        images: list[tuple[ImageReadResult, bool]] = []
        msg_with_image = QQMessage()
        for seg in message:
            if seg.type in {"image", "meme"}:
                if not seg.data.get("content"):
                    if image_data := seg.data.get("image"):
                        image_data = cast("ImageReadResult", image_data)
                        images.append((image_data, seg.type == "meme"))
                        msg_with_image += seg
                    elif image_file := seg.data.get("file"):
                        image_data = await self.get_image(image_file)
                        if image_data:
                            images.append((image_data, seg.type == "meme"))
                            msg_with_image += seg
                else:
                    msg_with_image += seg
            else:
                msg_with_image += seg

        msg = (
            UserMessage(**(msg_data | {"message": msg_with_image, "images": images}))
            if images
            else UserMessage(**msg_data)
        )

        return GroupEvent(
            session_id=session_id,
            history_storage=history_storage,
            message=msg,
        )

    @override
    async def handle(self) -> None:
        try:
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
        except Exception:
            logger.exception("Failed to handle group event")
        finally:
            self.stop()

    @override
    async def rule(self) -> bool:
        return str(self.event.user_id) != "2830758180"
