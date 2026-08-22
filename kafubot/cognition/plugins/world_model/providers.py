from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

import anyio
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from kafubot.agency.models import SelfState, SocialHome
    from kafubot.cognition.plugins.environment import SocialEnvironment


class ProviderQuery(BaseModel):
    operation: Literal["person", "media", "memory", "context"]
    session_id: str = ""
    user_id: str = ""
    message_id: str = ""
    query: str = ""
    limit: int = Field(default=12, ge=1, le=100)


class ProviderCommit(BaseModel):
    operation: Literal["reply", "skip", "clear"]
    session_id: str
    payload: dict[str, object] = Field(default_factory=dict)


class WorldModelProvider(Protocol):
    """Read/read-more/write boundary for plugin-provided world knowledge."""

    name: str

    async def peek(self, home: SocialHome, state: SelfState) -> str | None: ...

    async def query(self, request: ProviderQuery) -> str | None: ...

    async def commit(self, change: ProviderCommit) -> None: ...

    async def aclose(self) -> None: ...


class WorldModelHub:
    def __init__(self, providers: list[WorldModelProvider] | None = None) -> None:
        self.providers = providers or []

    async def peek(self, home: SocialHome, state: SelfState) -> list[str]:
        results: list[str | None] = [None] * len(self.providers)

        async def run(index: int, provider: WorldModelProvider) -> None:
            results[index] = await provider.peek(home, state)

        async with anyio.create_task_group() as task_group:
            for index, provider in enumerate(self.providers):
                task_group.start_soon(run, index, provider)
        return [result for result in results if result]

    async def query(self, request: ProviderQuery) -> list[str]:
        results: list[str | None] = [None] * len(self.providers)

        async def run(index: int, provider: WorldModelProvider) -> None:
            results[index] = await provider.query(request)

        async with anyio.create_task_group() as task_group:
            for index, provider in enumerate(self.providers):
                task_group.start_soon(run, index, provider)
        return [result for result in results if result]

    async def commit(self, change: ProviderCommit) -> None:
        async with anyio.create_task_group() as task_group:
            for provider in self.providers:
                task_group.start_soon(provider.commit, change)

    async def aclose(self) -> None:
        async with anyio.create_task_group() as task_group:
            for provider in self.providers:
                task_group.start_soon(provider.aclose)


class ConversationWorldProvider:
    """First provider: derives people, media and searchable memories from history."""

    name = "conversation"

    def __init__(self, environment: SocialEnvironment) -> None:
        self.environment = environment

    async def peek(self, home: SocialHome, state: SelfState) -> str | None:
        _ = state
        directed = sum(item.directed_count for item in home.items)
        if not home.items:
            return None
        return f"{len(home.items)} active conversations; {directed} direct messages"

    async def query(self, request: ProviderQuery) -> str | None:
        read_limit = (
            self.environment.max_entries
            if request.operation in {"memory", "media"}
            else request.limit
        )
        entries = await self.environment.read(
            request.session_id,
            limit=read_limit,
        )
        if request.operation == "person":
            matched = [item for item in entries if item.user_id == request.user_id]
            if not matched:
                return None
            lines = [f"{item.user}: {item.content}" for item in matched[-6:]]
            return "Recent observable messages from this person:\n" + "\n".join(lines)
        if request.operation == "media":
            matched = next(
                (item for item in entries if item.message_id == request.message_id),
                None,
            )
            if matched is None or not matched.has_media:
                return None
            return f"Observable media description: {matched.content}"
        if request.operation == "memory":
            needle = request.query.casefold().strip()
            matched = [
                item
                for item in entries
                if not needle or needle in item.content.casefold()
            ][-request.limit :]
            if not matched:
                return None
            return "Matching observable history:\n" + "\n".join(
                f"{item.user or 'agent'}: {item.content}" for item in matched
            )
        if request.operation == "context":
            return "Recent conversation context:\n" + "\n".join(
                f"{item.user or 'agent'}: {item.content}" for item in entries
            )
        return None

    async def commit(self, change: ProviderCommit) -> None:
        _ = change

    async def aclose(self) -> None:
        return
