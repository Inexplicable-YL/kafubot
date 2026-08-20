from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from kafubot.adapters.cqhttp.event import GroupMessageEvent, PrivateMessageEvent
    from kafubot.config import AgentConfig


class AgentRuntime(Protocol):
    async def run(self) -> None: ...

    async def handle(
        self,
        event: GroupMessageEvent | PrivateMessageEvent,
    ) -> None: ...

    async def aclose(self) -> None: ...


def create_agent_runtime(config: AgentConfig) -> AgentRuntime:
    from kafubot.social import SocialAgentRuntime  # noqa: PLC0415

    return SocialAgentRuntime(config)


__all__ = ["AgentRuntime", "create_agent_runtime"]
