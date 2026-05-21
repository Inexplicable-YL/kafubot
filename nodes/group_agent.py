import json
import math
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from typing_extensions import override
from zoneinfo import ZoneInfo

import anyio
import opencc
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import stats  # type: ignore[import-untyped]
from sekaibot import Node
from sekaibot.adapter.cqhttp.event import GroupMessageEvent
from sekaibot.config import ConfigModel

from chat.activity import ActivityStore, get_activity_store
from chat.agent import UserMessage, clear_session_history, get_agent
from chat.image import (
    ImageReadResult,
    get_image_analyzer,
    read_image,
)
from chat.meme import add_memes
from chat.message import QQMessage, QQMessageSegment

MESSAGES_LIMIT = 30
BACKUP_MESSAGES_LIMIT = 20
BACKUP_PENDING_COUNTS_LIMIT = 10

EXTRA_PROMPT_MAX_HISTORY = 5

DEFAULT_USERNAME = "陌生用户"

DEFAULT_CLEAR_KEYWORDS = {"/clear", "/清除"}
AGENT_DEBUG_LOG = Path(".database/group_agent_debug.jsonl")
DEBUG_TEXT_LIMIT = 500
DEBUG_MAX_DEPTH = 4


def _extract_stop_message(agent_output: Any) -> dict[str, Any] | None:
    candidates = [agent_output]
    while candidates:
        candidate = candidates.pop(0)
        if (
            isinstance(candidate, dict)
            and isinstance(candidate.get("type"), str)
            and isinstance(candidate.get("data"), dict)
        ):
            return candidate
        if isinstance(candidate, dict):
            candidates.extend(candidate.get(key) for key in ("output", "stop_message"))
            candidates.extend(candidate.values())
    return None


def _compact_text(text: Any, limit: int = DEBUG_TEXT_LIMIT) -> str:
    compacted = " ".join(str(text).split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3] + "..."


def _summarize_user_message(message: UserMessage) -> dict[str, Any]:
    return {
        "role": message.role,
        "time": message.timestamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "user": message.user,
        "user_id": message.user_id,
        "message_id": message.message_id,
        "is_tome": message.is_tome,
        "to_other": message.to_other,
        "have_keywords": message.have_keywords,
        "image_count": len(message.images),
        "text": _compact_text(message.message.get_msgcode()),
    }


def _summarize_debug_value(value: Any, *, depth: int = 0) -> Any:  # noqa: PLR0911
    if depth > DEBUG_MAX_DEPTH:
        return _compact_text(type(value).__name__, 80)
    if isinstance(value, UserMessage):
        return _summarize_user_message(value)
    if isinstance(value, QQMessage):
        return _compact_text(value.get_msgcode())
    if isinstance(value, str | int | float | bool) or value is None:
        return value if not isinstance(value, str) else _compact_text(value)
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in {"node", "image", "base64", "phash"}:
                output[key_text] = f"<{type(item).__name__}>"
                continue
            output[key_text] = _summarize_debug_value(item, depth=depth + 1)
        return output
    if isinstance(value, list | tuple | deque):
        return [
            _summarize_debug_value(item, depth=depth + 1) for item in list(value)[:20]
        ]
    if hasattr(value, "type") and hasattr(value, "content"):
        summary: dict[str, Any] = {
            "message_type": type(value).__name__,
            "content": _compact_text(getattr(value, "content", "")),
        }
        if tool_calls := getattr(value, "tool_calls", None):
            summary["tool_calls"] = [
                {
                    "name": call.get("name"),
                    "args": _summarize_debug_value(call.get("args"), depth=depth + 1),
                    "id": call.get("id"),
                }
                for call in tool_calls
                if isinstance(call, dict)
            ]
        return summary
    return f"<{type(value).__name__}>"


async def _write_agent_debug(event: dict[str, Any]) -> None:
    AGENT_DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "logged_at": datetime.now(tz=UTC).isoformat(),
        **event,
    }
    async with await anyio.open_file(AGENT_DEBUG_LOG, "a", encoding="utf-8") as file:
        await file.write(json.dumps(event, ensure_ascii=False) + "\n")


def _ttest_signal(statistic: Any) -> float:
    value = float(statistic)
    if math.isnan(value):
        return 0.0
    return -(math.tanh(value) if value > 0 else value)


class GroupAgentConfig(ConfigModel):
    __config_name__ = "group_agent"

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


class Histories(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: deque[UserMessage] = Field(
        default_factory=lambda: deque(maxlen=MESSAGES_LIMIT)
    )
    backup_messages: deque[UserMessage] = Field(
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
    message: UserMessage

    @property
    def time(self) -> int:
        return int(self.message.timestamp.timestamp())


class GroupAgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: Any = Field(default_factory=get_agent)
    image_analyzer: Runnable[dict[str, Any], str] = Field(
        default_factory=lambda _: get_image_analyzer(True, add_memes_hook=add_memes)
    )
    activity_store: ActivityStore = Field(default_factory=get_activity_store)
    storages: dict[str, Histories] = Field(default_factory=dict)
    storages_lock: anyio.Lock = Field(default_factory=anyio.Lock)


class GroupAgent(Node[GroupMessageEvent, GroupAgentState, GroupAgentConfig]):
    """群聊记录节点"""

    priority = 0
    block = True
    load = False

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
            len_messages = len(history_storage.messages)
            history_storage.backup_messages.append(message)
            if group_event.message.images:
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
                    "detail": False,
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

        print(f"Invoking-Group-Agent: {[m.message for m in current_messages]}")
        run_id = (
            f"{group_event.session_id}-"
            f"{group_event.message.message_id or group_event.time}-"
            f"{datetime.now(tz=UTC).strftime('%Y%m%d%H%M%S%f')}"
        )
        agent_input = {
            "messages": [],
            "inputs": current_messages,
            "early_messages": [],
            "full_messages": [],
            "reasoning_effort": "max",
            "should_stop": False,
            "stop_message": None,
        }
        agent_context = {
            "session_id": group_event.session_id,
            "node": self,
            "is_tome": group_event.message.is_tome,
        }
        await _write_agent_debug(
            {
                "run_id": run_id,
                "event": "start",
                "session_id": group_event.session_id,
                "current_messages": _summarize_debug_value(current_messages),
            }
        )

        agent_output: Any = None
        chunk_index = 0
        try:
            async for chunk in self.node_state.agent.astream(
                agent_input,
                context=agent_context,
                stream_mode="updates",
            ):
                agent_output = chunk
                await _write_agent_debug(
                    {
                        "run_id": run_id,
                        "event": "chunk",
                        "chunk_index": chunk_index,
                        "chunk": _summarize_debug_value(chunk),
                        "stop_message": _summarize_debug_value(
                            _extract_stop_message(chunk)
                        ),
                    }
                )
                chunk_index += 1
        except Exception as exc:
            await _write_agent_debug(
                {
                    "run_id": run_id,
                    "event": "error",
                    "error_type": type(exc).__name__,
                    "error": _compact_text(exc),
                }
            )
            raise

        stop_message = _extract_stop_message(agent_output)
        full_text = ""
        if stop_message and stop_message["type"] == "reply":
            full_text = str(stop_message["data"].get("full_text") or "").strip()
        await _write_agent_debug(
            {
                "run_id": run_id,
                "event": "end",
                "chunks": chunk_index,
                "stop_message": _summarize_debug_value(stop_message),
                "full_text": _compact_text(full_text),
            }
        )

        if full_text:
            group_event.history_storage.backup_messages.append(
                UserMessage(
                    role="assistant",
                    timestamp=datetime.now(tz=UTC),
                    user="可不",
                    message=QQMessage.from_str(full_text),
                    user_id=str(self.event.adapter.self_id),
                    message_id="",
                    is_tome=False,
                    to_other=False,
                    have_keywords=False,
                )
            )
            group_event.history_storage.backup_pending_counts.append(
                len(current_messages)
            )
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
    ) -> tuple[QQMessage | None, bool]:
        if (text := self.event.message.get_plain_text()) and any(
            keyw in text for keyw in self.config.clear_keywords
        ):
            await self.delete_chat(session_id=session_id)
            return None, False
        message = await QQMessage.from_cqhttp_message(
            self.event.message, history_storage.backup_messages, self.get_image
        )

        to_other = (not to_me) and any(seg.type == "at" for seg in self.event.message)
        if to_me:
            message = QQMessageSegment.at("可不") + message
        if self.event.reply and (reply_time := int(self.event.reply.time)):
            time_text = (
                datetime.fromtimestamp(reply_time, tz=UTC)
                .astimezone(ZoneInfo("Asia/Shanghai"))
                .strftime("%Y-%m-%d %H:%M:%S")
            )
            message = QQMessageSegment.reply(time_text) + message
            to_other = to_other or (
                self.event.reply.sender.user_id != self.event.adapter.self_id
            )
        return message, to_other

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
        if not message:
            return None
        timestamp = datetime.fromtimestamp(self.event.time, tz=UTC)
        user = self.event.sender.nickname or DEFAULT_USERNAME
        have_keywords = self.have_keywords(message.get_plain_text())
        if self.config.reply_when_keywords:
            is_tome = is_tome or have_keywords
        msg_data = {
            "role": "user",
            "timestamp": timestamp,
            "user": user,
            "message": message,
            "user_id": str(self.event.user_id),
            "message_id": str(self.event.message_id),
            "is_tome": is_tome,
            "to_other": to_other,
            "have_keywords": have_keywords,
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
        return (
            str(self.event.user_id) != "2830758180" and self.event.group_id == 895484096  # noqa: PLR2004
        )
