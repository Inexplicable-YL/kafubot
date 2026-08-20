from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

if TYPE_CHECKING:
    from kafubot.actions import QQActions
    from kafubot.agency.models import (
        ActionContract,
        CompiledContext,
        ConversationCandidate,
        ReplyResult,
        TimelineEntry,
    )
    from kafubot.cognition.types import UserMessage


def user_model_message(message: UserMessage) -> HumanMessage:
    """Expose a stable model message directly on the plugin event API."""
    return HumanMessage(
        content=message.as_content(),
        additional_kwargs={"raw": message},
    )


def timeline_model_message(entry: TimelineEntry) -> BaseMessage:
    """Convert observable timeline data without reconstructing a legacy runtime."""
    if entry.role == "assistant":
        return AIMessage(
            content=entry.content,
            additional_kwargs={"created_at": entry.timestamp},
        )
    if entry.raw is not None:
        return user_model_message(entry.raw)
    return HumanMessage(content=entry.content)


@dataclass(slots=True, frozen=True)
class ObservationEvent:
    session_id: str
    message: UserMessage
    candidate: ConversationCandidate
    history: tuple[TimelineEntry, ...]
    actions: QQActions

    @property
    def model_messages(self) -> tuple[BaseMessage, ...]:
        return (user_model_message(self.message),)

    @property
    def history_messages(self) -> tuple[BaseMessage, ...]:
        return tuple(timeline_model_message(entry) for entry in self.history)


@dataclass(slots=True)
class ReplyPreparation:
    contract: ActionContract
    context: CompiledContext
    actions: QQActions
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def user_messages(self) -> tuple[UserMessage, ...]:
        return tuple(
            entry.raw
            for entry in self.context.messages
            if entry.role == "user" and entry.raw is not None
        )

    @property
    def model_messages(self) -> tuple[BaseMessage, ...]:
        return tuple(timeline_model_message(entry) for entry in self.context.messages)


@dataclass(slots=True, frozen=True)
class ReplyCommitted:
    preparation: ReplyPreparation
    result: ReplyResult

    @property
    def model_messages(self) -> tuple[BaseMessage, ...]:
        return (
            *self.preparation.model_messages,
            AIMessage(content=self.result.full_text),
        )


@dataclass(slots=True, frozen=True)
class SkipCommitted:
    session_id: str
    reason: str
    history: tuple[TimelineEntry, ...]


__all__ = [
    "ObservationEvent",
    "ReplyCommitted",
    "ReplyPreparation",
    "SkipCommitted",
    "timeline_model_message",
    "user_model_message",
]
