from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

EventReceiver = Callable[["Event"], Awaitable[None]]


class Event(BaseModel):
    """Protocol-neutral event base class."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    adapter: Any = Field(exclude=True)

    @property
    @abstractmethod
    def event_name(self) -> str: ...


__all__ = ["Event", "EventReceiver"]
