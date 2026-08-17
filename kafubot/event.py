from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Generic, NamedTuple, TypeVar
from typing_extensions import override

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from kafubot.message import Message

AdapterT = TypeVar("AdapterT")
EventReceiver = Callable[["Event[Any]"], Awaitable[None]]


class Event(ABC, BaseModel, Generic[AdapterT]):
    """Adapter-native event base, compatible with SekaiBot's internal event API."""

    model_config = ConfigDict(extra="allow")
    adapter: Any = Field(exclude=True)
    type: str | None
    __handled__: bool = False

    @property
    def event_name(self) -> str:
        """Compatibility property used by KafuBot's transport logging."""
        return self.get_event_name()

    @override
    def __str__(self) -> str:
        return f"Event<{self.get_event_name()}>\n{self.get_event_description()}"

    @override
    def __repr__(self) -> str:
        return self.__str__()

    def get_event_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def get_event_description(self) -> str:
        raise NotImplementedError

    def get_log_string(self) -> str:
        return str(self)

    @abstractmethod
    def get_user_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_session_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_message(self) -> Message[Any]:
        raise NotImplementedError

    def get_plain_text(self) -> str:
        return self.get_message().get_plain_text()

    @abstractmethod
    def get_conversation_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def is_tome(self) -> bool:
        raise NotImplementedError


class EventHandleOption(NamedTuple):
    """Event dispatch metadata retained for adapter-level event consumers."""

    event: Event[Any]
    handle_get: bool


__all__ = ["Event", "EventHandleOption", "EventReceiver"]
