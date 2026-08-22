from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict, Field

from kafubot.actions import QQActions  # noqa: TC001 - Pydantic runtime type
from kafubot.agency.models import (
    ActionContract,  # noqa: TC001 - Pydantic runtime type
    CompiledContext,  # noqa: TC001 - Pydantic runtime type
    ConversationCandidate,  # noqa: TC001 - Pydantic runtime type
    ReplyResult,  # noqa: TC001 - Pydantic runtime type
    TimelineEntry,  # noqa: TC001 - Pydantic runtime type
)
from kafubot.cognition.types import UserMessage  # noqa: TC001 - Pydantic runtime type


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


class LifecycleEvent(BaseModel):
    """Validated data passed through plugin lifecycle hooks."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)


class ObservationEvent(LifecycleEvent):
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


class ContextWindowEvicted(LifecycleEvent):
    """Messages that just transitioned out of a session's live model window."""

    session_id: str
    entries: tuple[TimelineEntry, ...]

    @property
    def model_messages(self) -> tuple[BaseMessage, ...]:
        return tuple(timeline_model_message(entry) for entry in self.entries)


class ReplyPreparation(LifecycleEvent):
    model_config = ConfigDict(frozen=False)

    contract: ActionContract
    context: CompiledContext
    actions: QQActions
    metadata: dict[str, Any] = Field(default_factory=dict)

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


class ReplyCommitted(LifecycleEvent):
    preparation: ReplyPreparation
    result: ReplyResult

    @property
    def model_messages(self) -> tuple[BaseMessage, ...]:
        return (
            *self.preparation.model_messages,
            AIMessage(content=self.result.full_text),
        )


class SkipCommitted(LifecycleEvent):
    session_id: str
    reason: str
    history: tuple[TimelineEntry, ...]


__all__ = [
    "ContextWindowEvicted",
    "LifecycleEvent",
    "ObservationEvent",
    "ReplyCommitted",
    "ReplyPreparation",
    "SkipCommitted",
    "timeline_model_message",
    "user_model_message",
]
