import contextlib
import json
import os
from dataclasses import dataclass, field
from typing import Any
from typing_extensions import override

import aiohttp
import anyio
from sekaibot import Node
from sekaibot.adapter.cqhttp.event import PrivateMessageEvent

from dify_client import AsyncChatClient

_api_key = os.environ.get("DIFY_API_KEY") or "app-GoBKws26waHSs9XpWIYSdU9A"
if _api_key is None:
    raise ValueError("DIFY_API_KEY is not set")

API_KEY = _api_key
BASE_URL = "http://192.168.3.55:433/v1"


@dataclass
class EnqueueDecision:
    run_now: bool
    waiter_id: int


@dataclass
class SessionQueueState:
    running: bool = False
    pending_messages: list[str] = field(default_factory=list)
    latest_waiter_id: int = 0
    condition: anyio.Condition = field(default_factory=anyio.Condition)

    async def enqueue(self, message: str) -> EnqueueDecision:
        async with self.condition:
            if not self.running and not self.pending_messages:
                self.running = True
                return EnqueueDecision(run_now=True, waiter_id=0)

            self.pending_messages.append(message)
            self.latest_waiter_id += 1
            waiter_id = self.latest_waiter_id
            self.condition.notify_all()
            return EnqueueDecision(run_now=False, waiter_id=waiter_id)

    async def wait_and_claim(self, waiter_id: int) -> str | None:
        async with self.condition:
            while True:
                if waiter_id != self.latest_waiter_id:
                    return None

                if not self.running and self.pending_messages:
                    merged_message = "\n\n".join(self.pending_messages)
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
    sessions: dict[str, SessionQueueState] = field(default_factory=dict)
    sessions_lock: anyio.Lock = field(default_factory=anyio.Lock)


class PrivateReply(Node[PrivateMessageEvent, PrivateReplyState, Any]):  # type: ignore
    priority = 1

    @override
    def __init_state__(self) -> PrivateReplyState:
        return PrivateReplyState()

    async def _get_session_state(self, session_id: str) -> SessionQueueState:
        async with self.node_state.sessions_lock:
            if session_id not in self.node_state.sessions:
                self.node_state.sessions[session_id] = SessionQueueState()
            return self.node_state.sessions[session_id]

    async def _delete_conversation(self, conversation_id: str, user: str) -> None:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        }
        payload = {"user": user}
        url = f"{BASE_URL}/conversations/{conversation_id}"
        timeout = aiohttp.ClientTimeout(total=180)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.delete(url, headers=headers, json=payload) as response,
        ):
            response.raise_for_status()

    async def _run_chat(self, query: str, name: str, session_id: str) -> None:
        async with AsyncChatClient(
            API_KEY,
            base_url=BASE_URL,
            timeout=180,
        ) as client:
            answer: str = ""
            response = await client.create_chat_message(
                inputs={"name": name},
                query=query,
                user=session_id,
                conversation_id=session_id,
                response_mode="streaming",
            )
            response.raise_for_status()
            async for segment in response.aiter_lines():
                if segment.startswith("data:") and (data := segment[5:].strip()):
                    with contextlib.suppress(json.JSONDecodeError):
                        if chunk := json.loads(data):
                            event: str = chunk.get("event")
                            if event == "message":
                                text: str = chunk.get("answer", "").strip()
                                if text:
                                    lines = text.split("\n")
                                    for i, line in enumerate(lines):
                                        stripped = line.strip()
                                        if i < len(lines) - 1:
                                            answer += stripped
                                            if answer:
                                                print(answer)
                                                await self.reply(answer)
                                            answer = ""
                                        else:
                                            answer += stripped
                            elif event == "message_end":
                                break

            if answer.strip():
                await self.reply(answer.strip())

    @override
    async def handle(self) -> None:
        text = self.event.get_plain_text()
        if not text:
            return
        session_id = self.event.get_session_id()

        keyws = ["clear", "清除", "清空", "清理", "删除", "重置", "重新开始", "重启"]
        if any(keyw in text for keyw in keyws):
            await self._delete_conversation(
                conversation_id=session_id,
                user=session_id,
            )
            return

        name_map: dict[str, str] = {
            "shiroko": "白子小姐",
            "空想少女": "アルス",
            "かたちなきもの": "言霊",
        }
        name: str = self.event.sender.nickname or ""
        name = name_map.get(name, name)

        session_state = await self._get_session_state(session_id)
        decision = await session_state.enqueue(text)

        if decision.run_now:
            query = text
        else:
            queued_query = await session_state.wait_and_claim(decision.waiter_id)
            if queued_query is None:
                return
            query = queued_query

        try:
            await self._run_chat(query=query, name=name, session_id=session_id)
        finally:
            await session_state.finish_run()
