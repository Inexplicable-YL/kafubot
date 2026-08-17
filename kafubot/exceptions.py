from __future__ import annotations

from typing import Any

__all__ = [
    "AdapterException",
    "GetEventTimeout",
    "KafuBotException",
    "MockApiException",
]


class KafuBotException(Exception):  # noqa: N818 - public compatibility name
    pass


class GetEventTimeout(KafuBotException):
    pass


class AdapterException(KafuBotException):
    pass


class MockApiException(KafuBotException):
    def __init__(self, result: Any) -> None:
        self.result = result
        super().__init__(f"mocked API result: {result!r}")
