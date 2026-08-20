from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import anyio
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from kafubot.agency.models import ConversationCandidate, TimelineEntry
from kafubot.cognition.history import clear_session_history, get_session_history
from kafubot.cognition.plugins.base import PluginContext, PluginDefinition
from kafubot.cognition.types import UserMessage
from kafubot.cognition.utils import content_to_text

if TYPE_CHECKING:
    from kafubot.actions import QQActions
    from kafubot.config import AgentConfig

logger = logging.getLogger(__name__)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class HistoryRepository(Protocol):
    async def load(self, session_id: str) -> list[TimelineEntry]: ...

    async def append_user(self, session_id: str, message: UserMessage) -> None: ...

    async def append_assistant(self, session_id: str, content: str) -> None: ...

    async def clear(self, session_id: str) -> None: ...


class AgentHistoryRepository:
    def __init__(self) -> None:
        Path(".database").mkdir(parents=True, exist_ok=True)

    async def load(self, session_id: str) -> list[TimelineEntry]:
        messages = await get_session_history(session_id).aget_messages()
        return [
            entry
            for message in messages
            if (entry := self._to_entry(message)) is not None
        ]

    async def append_user(self, session_id: str, message: UserMessage) -> None:
        await get_session_history(session_id).aadd_message(
            HumanMessage(
                content=message.as_content(),
                additional_kwargs={"raw": message},
            )
        )

    async def append_assistant(self, session_id: str, content: str) -> None:
        await get_session_history(session_id).aadd_message(AIMessage(content=content))

    async def clear(self, session_id: str) -> None:
        await clear_session_history(session_id)

    @staticmethod
    def _to_entry(message: BaseMessage) -> TimelineEntry | None:
        if isinstance(message, HumanMessage):
            raw = message.additional_kwargs.get("raw")
            if isinstance(raw, UserMessage):
                return TimelineEntry(
                    role="user",
                    timestamp=_aware_utc(raw.timestamp),
                    content=raw.message.get_msgcode(),
                    user=raw.user,
                    user_id=raw.user_id,
                    message_id=raw.message_id,
                    directed_to_bot=raw.is_tome,
                    has_media=bool(raw.images),
                    raw=raw,
                )
            return TimelineEntry(
                role="user",
                timestamp=datetime.now(UTC),
                content=content_to_text(message.content),
            )
        if isinstance(message, AIMessage):
            timestamp = message.additional_kwargs.get("created_at")
            if not isinstance(timestamp, datetime):
                timestamp = datetime.now(UTC)
            return TimelineEntry(
                role="assistant",
                timestamp=_aware_utc(timestamp),
                content=content_to_text(message.content),
            )
        return None


class InMemoryHistoryRepository:
    """Small repository used by tests and local embeddings of the runtime."""

    def __init__(self) -> None:
        self.entries: dict[str, list[TimelineEntry]] = {}

    async def load(self, session_id: str) -> list[TimelineEntry]:
        return [item.model_copy(deep=True) for item in self.entries.get(session_id, [])]

    async def append_user(self, session_id: str, message: UserMessage) -> None:
        self.entries.setdefault(session_id, []).append(
            TimelineEntry(
                role="user",
                timestamp=message.timestamp,
                content=message.message.get_msgcode(),
                user=message.user,
                user_id=message.user_id,
                message_id=message.message_id,
                directed_to_bot=message.is_tome,
                has_media=bool(message.images),
                raw=message,
            )
        )

    async def append_assistant(self, session_id: str, content: str) -> None:
        self.entries.setdefault(session_id, []).append(
            TimelineEntry(
                role="assistant",
                timestamp=datetime.now(UTC),
                content=content,
            )
        )

    async def clear(self, session_id: str) -> None:
        self.entries.pop(session_id, None)


class ConversationState:
    """One context partition inside the shared social environment."""

    def __init__(self, max_entries: int) -> None:
        self.entries: deque[TimelineEntry] = deque(maxlen=max_entries)
        self.latest_actions: QQActions | None = None
        self.last_handled_sequence = 0
        self.last_read_sequence = 0
        self.hydrated = False

    @property
    def latest_sequence(self) -> int:
        return self.entries[-1].sequence if self.entries else 0

    @property
    def unread(self) -> list[TimelineEntry]:
        return [
            item
            for item in self.entries
            if item.role == "user" and item.sequence > self.last_handled_sequence
        ]


class SocialEnvironment:
    """Observable world state shared by one executive across all conversations."""

    def __init__(
        self,
        *,
        history: HistoryRepository | None = None,
        max_entries: int = 100,
    ) -> None:
        self.history = history or AgentHistoryRepository()
        self.max_entries = max_entries
        self.sessions: dict[str, ConversationState] = {}
        self._sequence = 0
        self._lock = anyio.Lock()

    async def session(self, session_id: str) -> ConversationState:
        async with self._lock:
            state = self.sessions.setdefault(
                session_id,
                ConversationState(self.max_entries),
            )
            if not state.hydrated:
                stored = await self.history.load(session_id)
                for entry in stored[-self.max_entries :]:
                    self._sequence += 1
                    entry.sequence = self._sequence
                    state.entries.append(entry)
                state.last_handled_sequence = state.latest_sequence
                state.last_read_sequence = state.latest_sequence
                state.hydrated = True
            return state

    async def observe(
        self,
        session_id: str,
        message: UserMessage,
        actions: QQActions,
    ) -> ConversationCandidate:
        state = await self.session(session_id)
        async with self._lock:
            self._sequence += 1
            state.entries.append(
                TimelineEntry(
                    sequence=self._sequence,
                    role="user",
                    timestamp=message.timestamp,
                    content=message.message.get_msgcode(),
                    user=message.user,
                    user_id=message.user_id,
                    message_id=message.message_id,
                    directed_to_bot=message.is_tome,
                    has_media=bool(message.images),
                    raw=message,
                )
            )
            state.latest_actions = actions
            candidate = self._candidate(session_id, state)
        try:
            await self.history.append_user(session_id, message)
        except Exception:
            logger.exception("Failed to persist observed social event")
        return candidate

    async def candidates(self) -> list[ConversationCandidate]:
        async with self._lock:
            return [
                self._candidate(session_id, state)
                for session_id, state in self.sessions.items()
                if state.unread
            ]

    async def read(
        self,
        session_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[TimelineEntry]:
        state = await self.session(session_id)
        async with self._lock:
            entries = list(state.entries)
            if before_sequence is not None:
                entries = [item for item in entries if item.sequence < before_sequence]
            selected = entries[-limit:]
            if selected:
                state.last_read_sequence = max(
                    state.last_read_sequence,
                    selected[-1].sequence,
                )
            return [item.model_copy(deep=True) for item in selected]

    async def find_message(
        self,
        session_id: str,
        message_id: str,
    ) -> TimelineEntry | None:
        state = await self.session(session_id)
        async with self._lock:
            return next(
                (
                    item.model_copy(deep=True)
                    for item in reversed(state.entries)
                    if item.message_id == message_id
                ),
                None,
            )

    async def entries_by_sequence(
        self,
        session_id: str,
        sequences: set[int],
    ) -> list[TimelineEntry]:
        state = await self.session(session_id)
        async with self._lock:
            return [
                item.model_copy(deep=True)
                for item in state.entries
                if item.sequence in sequences
            ]

    async def actions_for(self, session_id: str) -> QQActions | None:
        state = await self.session(session_id)
        async with self._lock:
            return state.latest_actions

    async def mark_handled(
        self,
        session_id: str,
        *,
        through_sequence: int | None = None,
    ) -> None:
        state = await self.session(session_id)
        async with self._lock:
            state.last_handled_sequence = max(
                state.last_handled_sequence,
                through_sequence or state.latest_sequence,
            )

    async def commit_reply(
        self,
        session_id: str,
        content: str,
        *,
        handled_through_sequence: int,
    ) -> None:
        state = await self.session(session_id)
        now = datetime.now(UTC)
        async with self._lock:
            self._sequence += 1
            state.entries.append(
                TimelineEntry(
                    sequence=self._sequence,
                    role="assistant",
                    timestamp=now,
                    content=content,
                )
            )
            state.last_handled_sequence = max(
                state.last_handled_sequence,
                handled_through_sequence,
            )
        try:
            await self.history.append_assistant(session_id, content)
        except Exception:
            logger.exception("Failed to persist assistant message")

    async def clear(self, session_id: str) -> None:
        await self.history.clear(session_id)
        async with self._lock:
            self.sessions.pop(session_id, None)

    async def has_unhandled(self) -> bool:
        async with self._lock:
            return any(state.unread for state in self.sessions.values())

    @staticmethod
    def _candidate(
        session_id: str,
        state: ConversationState,
    ) -> ConversationCandidate:
        unread = state.unread
        latest = unread[-1] if unread else state.entries[-1]
        people = list(
            dict.fromkeys(
                item.user
                for item in reversed(state.entries)
                if item.role == "user" and item.user
            )
        )[:5]
        preview = latest.content.replace("\n", " ").strip()
        if len(preview) > 120:
            preview = f"{preview[:117]}..."
        return ConversationCandidate(
            session_id=session_id,
            title=session_id,
            preview=preview,
            people=people,
            latest_at=latest.timestamp,
            new_count=len(unread),
            directed_count=sum(item.directed_to_bot for item in unread),
            question_count=sum(
                "?" in item.content or "？" in item.content for item in unread
            ),
            latest_sequence=state.latest_sequence,
        )


def apply(context: PluginContext, _config: Any) -> None:
    if context.optional_service("environment") is not None:
        return
    config = cast("AgentConfig", context.service("config"))
    context.provide(
        "environment",
        SocialEnvironment(max_entries=config.environment_message_limit),
    )


plugin = PluginDefinition(name="environment", apply=apply)
