"""KafuBot runtime and protocol adapters."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bot import Bot

__all__ = ["Bot"]


def __getattr__(name: str) -> Any:
    if name == "Bot":
        from .bot import Bot  # noqa: PLC0415 - lazy import prevents adapter cycle

        return Bot
    raise AttributeError(name)
