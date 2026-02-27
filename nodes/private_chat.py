import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing_extensions import override

import aiosqlite
import anyio
import httpx
from sekaibot import Node
from sekaibot.adapter.cqhttp.event import PrivateMessageEvent

from dify_client import AsyncChatClient

_api_key = os.environ.get("DIFY_API_KEY")
if _api_key is None:
    raise ValueError("DIFY_API_KEY is not set")

API_KEY = _api_key
BASE_URL = os.environ.get("DIFY_BASE_URL")
CACHE_DIR = Path(".cache")
SESSION_DB_PATH = CACHE_DIR / "private_chat.db"
LOGGER = logging.getLogger(__name__)
MERGE_WINDOW_SECONDS = 5.0


@dataclass
class SessionConversationStore:
    db_path: Path = field(default_factory=lambda: SESSION_DB_PATH)
    init_lock: anyio.Lock = field(default_factory=anyio.Lock)
    initialized: bool = False

    async def ensure_initialized(self) -> None:
        if self.initialized:
            return

        async with self.init_lock:
            if self.initialized:
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_conversations (
                        session_id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        updated_at INTEGER NOT NULL
                    )
                    """
                )
                await db.commit()

            self.initialized = True

    async def get_conversation_id(self, session_id: str) -> str | None:
        await self.ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT conversation_id FROM session_conversations WHERE session_id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()

        if row is None:
            return None

        conversation_id = row[0]
        if not isinstance(conversation_id, str) or not conversation_id:
            return None
        return conversation_id

    async def set_conversation_id(self, session_id: str, conversation_id: str) -> None:
        await self.ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO session_conversations (session_id, conversation_id, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    conversation_id = excluded.conversation_id,
                    updated_at = excluded.updated_at
                """,
                (session_id, conversation_id, int(time.time())),
            )
            await db.commit()

    async def delete_conversation_id(self, session_id: str) -> None:
        await self.ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM session_conversations WHERE session_id = ?",
                (session_id,),
            )
            await db.commit()


@dataclass
class EnqueueDecision:
    waiter_id: int


@dataclass
class SessionQueueState:
    running: bool = False
    pending_messages: list[str] = field(default_factory=list)
    latest_waiter_id: int = 0
    last_enqueue_time: float | None = None
    merge_window_seconds: float = MERGE_WINDOW_SECONDS
    condition: anyio.Condition = field(default_factory=anyio.Condition)

    async def enqueue(self, message: str) -> EnqueueDecision:
        """Add a message to the queue.

        Note: Each new message resets the merge window timer, allowing messages
        arriving in quick succession to be batched together. The window extends
        from the most recent message, not the first.
        """
        async with self.condition:
            self.pending_messages.append(message)
            self.latest_waiter_id += 1
            self.last_enqueue_time = anyio.current_time()
            waiter_id = self.latest_waiter_id
            self.condition.notify_all()
            return EnqueueDecision(waiter_id=waiter_id)

    async def wait_and_claim(self, waiter_id: int) -> str | None:
        """Wait for messages to be ready for processing.

        Returns None if a newer message has superseded this waiter.
        Waits for the merge window to expire with no new messages before claiming.

        Note: Uses anyio.move_on_after for timeout, which may cancel the wait
        early. After timeout, we recalculate remaining time on next iteration.
        """
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
    conversation_store: SessionConversationStore = field(
        default_factory=SessionConversationStore
    )


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

    async def _delete_conversation(self, session_id: str) -> None:
        conversation_store = self.node_state.conversation_store
        conversation_id = await conversation_store.get_conversation_id(session_id)
        if conversation_id is None:
            return

        async with AsyncChatClient(
            API_KEY,
            base_url=BASE_URL,
            timeout=180,
        ) as client:
            try:
                response = await client.delete_conversation(
                    conversation_id=conversation_id,
                    user=session_id,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == httpx.codes.NOT_FOUND:
                    await conversation_store.delete_conversation_id(session_id)
                    return
                LOGGER.warning(
                    "Failed to delete Dify conversation for session_id=%s status=%s",
                    session_id,
                    exc.response.status_code,
                    exc_info=exc,
                )
                return
            except Exception as exc:
                LOGGER.warning(
                    "Failed to delete Dify conversation for session_id=%s",
                    session_id,
                    exc_info=exc,
                )
                return

        await conversation_store.delete_conversation_id(session_id)

    async def _run_chat(self, query: str, name: str, session_id: str) -> None:
        conversation_store = self.node_state.conversation_store
        conversation_id = await conversation_store.get_conversation_id(session_id) or ""
        latest_conversation_id: str | None = None

        async with AsyncChatClient(
            API_KEY,
            base_url=BASE_URL,
            timeout=180,
        ) as client:
            answer: str = ""
            LOGGER.debug("Processing query: %s", query)
            response = await client.create_chat_message(
                inputs={"name": name},
                query=query,
                user=session_id,
                conversation_id=conversation_id,
                response_mode="streaming",
            )
            response.raise_for_status()
            async for segment in response.aiter_lines():
                if segment.startswith("data:") and (data := segment[5:].strip()):
                    with contextlib.suppress(json.JSONDecodeError):
                        if chunk := json.loads(data):
                            chunk_conversation_id = chunk.get("conversation_id", "")
                            if (
                                isinstance(chunk_conversation_id, str)
                                and chunk_conversation_id.strip()
                            ):
                                latest_conversation_id = chunk_conversation_id.strip()
                            event: str = chunk.get("event")
                            if event == "message":
                                answer = await self._process_message_chunk(
                                    chunk, answer
                                )
                            elif event == "message_end":
                                break

            if answer.strip():
                await self.reply(answer.strip())
            if latest_conversation_id and latest_conversation_id != conversation_id:
                await conversation_store.set_conversation_id(
                    session_id, latest_conversation_id
                )

    async def _process_message_chunk(self, chunk: dict[str, Any], answer: str) -> str:
        """Process a streaming message chunk and send reply.

        Args:
            chunk: The parsed JSON chunk from the streaming response
            answer: Accumulated answer text

        Returns:
            Updated answer text (remaining buffer after sending chunks)
        """
        text: str = chunk.get("answer", "")
        LOGGER.debug("Received chunk: %s", text)
        text = text.strip(" ")
        if not text:
            return answer

        lines = text.split("\n")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i < len(lines) - 1:
                answer += stripped
                if answer:
                    LOGGER.debug("Sending reply chunk: %s", answer)
                    await self.reply(answer)
                answer = ""
            else:
                answer += stripped
        return answer

    @override
    async def handle(self) -> None:
        text = self.event.get_plain_text()
        if not text:
            return
        session_id = self.event.get_session_id()

        keyws = ["clear", "清除", "清空", "清理", "删除", "重置", "重新开始", "重启"]
        if any(keyw in text for keyw in keyws):
            await self._delete_conversation(session_id=session_id)
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

        query = await session_state.wait_and_claim(decision.waiter_id)
        if query is None:
            return

        try:
            await self._run_chat(query=query, name=name, session_id=session_id)
        finally:
            await session_state.finish_run()
