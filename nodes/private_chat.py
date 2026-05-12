import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from typing_extensions import override
from zoneinfo import ZoneInfo

import anyio
from _image import get_image_analyzer, read_image_as_base64
from _private_chat import (
    clear_session_history,
    get_chat_app,
)
from _prompt import get_extra_prompt
from langchain_core.runnables import Runnable
from pydantic import model_validator
from sekaibot import Node
from sekaibot.adapter.cqhttp.event import PrivateMessageEvent
from sekaibot.config import ConfigModel

BACKUP_MESSAGES_LIMIT = 10

EXTRA_PROMPT_MAX_HISTORY = 3


class PrivateChatConfig(ConfigModel):
    """私聊记录节点配置"""

    __config_name__ = "private_chat"
    merge_window_seconds: float = 5
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
        return self

    @property
    def activity_history_limit(self):
        return math.ceil(
            max((threshold for _, threshold in self.activity_limits), default=0)
            / min(self.activity_day_multiplier, self.activity_night_multiplier)
        )


@dataclass
class ActivityRecord:
    timestamp: datetime
    weight: float


@dataclass
class SessionQueueState:
    running: bool = False
    pending_messages: list[str] = field(default_factory=list)
    latest_waiter_id: int = 0
    last_enqueue_time: float | None = None
    merge_window_seconds: float = 5
    activites: list[ActivityRecord] = field(default_factory=list)
    condition: anyio.Condition = field(default_factory=anyio.Condition)

    async def enqueue(self, message: str) -> int:
        async with self.condition:
            self.pending_messages.append(message)
            self.latest_waiter_id += 1
            self.last_enqueue_time = anyio.current_time()
            waiter_id = self.latest_waiter_id
            self.condition.notify_all()
            return waiter_id

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

                    merged_message = self.pending_messages.copy()
                    self.pending_messages.clear()
                    self.running = True
                    return merged_message

                await self.condition.wait()

    async def finish_run(self) -> None:
        async with self.condition:
            self.running = False
            self.condition.notify_all()


@dataclass
class PrivateReplyState:
    chat_app: Runnable[dict[str, Any], str] = field(default_factory=get_chat_app)
    image_analyzer: Runnable[dict[str, Any], str] = field(
        default_factory=get_image_analyzer
    )
    backup_histories: dict[str, list[str]] = field(default_factory=dict)
    sessions: dict[str, SessionQueueState] = field(default_factory=dict)
    sessions_lock: anyio.Lock = field(default_factory=anyio.Lock)


class PrivateReply(Node[PrivateMessageEvent, PrivateReplyState, PrivateChatConfig]):  # type: ignore
    priority = 1

    def __init_state__(self) -> PrivateReplyState:
        return PrivateReplyState()

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

    def _is_activity_limited(
        self, activites: list[ActivityRecord], event_time: int
    ) -> bool:
        return any(
            threshold > 0
            and window_seconds > 0
            and sum(
                activity.weight
                for activity in activites
                if event_time - int(activity.timestamp.timestamp()) < window_seconds
            )
            >= threshold
            for window_seconds, threshold in self.config.activity_limits
        )

    async def _delete_chat(self, session_id: str) -> None:
        await clear_session_history(session_id)
        async with self.node_state.sessions_lock:
            if session_id in self.node_state.sessions:
                del self.node_state.sessions[session_id]
        await self.reply("[SYSTEM]已清除对话历史。")

    async def _run_chat(self, messages: list[str], name: str, session_id: str) -> bool:
        self.node_state.backup_histories.setdefault(session_id, [])
        self.node_state.backup_histories[session_id].append("\n".join(messages))
        replied = False
        answer = ""
        async for reply in self.node_state.chat_app.astream(
            {
                "messages": messages,
                "extra_prompt": get_extra_prompt(
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
                "reasoning_effort": "high",
            },
            config={"configurable": {"session_id": session_id}},
        ):
            if reply is not None:
                await self.reply(reply)
                answer += reply
                replied = True
        if replied:
            self.node_state.backup_histories[session_id].append(answer)
        self.node_state.backup_histories[session_id] = self.node_state.backup_histories[
            session_id
        ][-BACKUP_MESSAGES_LIMIT:]
        return replied

    async def get_image(self, file: str) -> str | None:
        try:
            result: dict[str, str] = await self.event.adapter.call_api(
                "get_image", file=file
            )
            path, url = result.get("file"), result.get("url")
            if path is not None:
                return await read_image_as_base64(path, url)
        except Exception:
            return None
        return None

    @override
    async def handle(self) -> None:
        text = self.event.get_plain_text()
        if not text:
            file: str | None = None
            if len(self.event.message) == 1 and self.event.message[0].type == "image":
                file = self.event.message[0].data.get("file")
            if file is not None:
                base64 = await self.get_image(file)
                print(
                    f"base64: {base64[:100]}..."
                    if base64
                    else "No image found or failed to read image."
                )
            return
        text = (
            f"[{datetime.fromtimestamp(self.event.time, tz=UTC).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')}]"
            + text
        )

        session_id = self.event.get_session_id()

        keyws = ["clear", "清除", "清空", "清理", "删除", "重置", "重新开始", "重启"]
        if any(keyw in text for keyw in keyws):
            await self._delete_chat(session_id=session_id)
            return

        name_map: dict[str, str] = {
            "shiroko": "白子小姐",
            "空想少女": "アルス",
            "かたちなきもの": "言霊",
        }
        name: str = self.event.sender.nickname or ""
        name = name_map.get(name, name)

        async with self.node_state.sessions_lock:
            if session_id not in self.node_state.sessions:
                self.node_state.sessions[session_id] = SessionQueueState(
                    merge_window_seconds=self.config.merge_window_seconds
                )
            session_state = self.node_state.sessions[session_id]

        if self._is_activity_limited(session_state.activites, self.event.time):
            return

        decision = await session_state.enqueue(text)

        messages = await session_state.wait_and_claim(decision)
        if messages is None:
            return

        try:
            replied = await self._run_chat(
                messages=messages,
                name=name,
                session_id=session_id,
            )
            if replied:
                session_state.activites.append(
                    ActivityRecord(
                        timestamp=datetime.fromtimestamp(self.event.time, tz=UTC),
                        weight=self._activity_weight(self.event.time),
                    )
                )
                if len(session_state.activites) > self.config.activity_history_limit:
                    session_state.activites = (
                        session_state.activites[-self.config.activity_history_limit :]
                        if self.config.activity_history_limit > 0
                        else []
                    )
        finally:
            await session_state.finish_run()
