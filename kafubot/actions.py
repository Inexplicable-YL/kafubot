from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kafubot.adapters.cqhttp.event import CQHTTPEvent
    from kafubot.adapters.cqhttp.message import CQHTTPMessageSegment
    from kafubot.message import BuildMessageType


class QQActions:
    """Per-event output context exposed to the cognitive layer."""

    def __init__(self, event: CQHTTPEvent) -> None:
        self.event = event

    async def reply(
        self,
        message: BuildMessageType[CQHTTPMessageSegment],
        **params: Any,
    ) -> Any:
        return await self.event.adapter.send(self.event, message, **params)

    async def call_api(self, api: str, **params: Any) -> Any:
        return await self.event.adapter.call_api(api, **params)
